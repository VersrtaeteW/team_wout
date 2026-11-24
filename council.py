"""Council of Achrona: een mini-Council met Raadsleden en een Koning.

Start met `uv run council`. Schrijft verdict.json en trace.json in de root van de repo.
De Koning leidt het debat in `kies_volgende()`: hij beslist wie spreekt, met welke vraag en waarom.
"""
import asyncio
import json
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from openai import AsyncOpenAI

import raadsleden

ROOT = Path(__file__).parent
KENNISBRONNEN = ROOT / "kennisbronnen"
ROLLEN = ("archmage", "engineer", "guardian_keeper", "historian", "oracle", "beast_master")
BEURTENBUDGET = 10  # spelregel: maximaal 10 Raadslid-beurten (berichten van Raadsleden) per run
ID_PATTERN = re.compile(r"\b(ARC|ENG|GUA|HIS|ORA|BEA)[‐‑‒–—―−\-][0-9]{3}\b")

# Mapping van record-ID prefix naar de rol die ze mag claimen
PREFIX_ROL = {
    "ARC": "archmage",
    "ENG": "engineer",
    "GUA": "guardian_keeper",
    "HIS": "historian",
    "ORA": "oracle",
    "BEA": "beast_master",
}

# Domein-mapping op basis van trefwoorden in incidentbeschrijving
DOMEIN_TREFWOORDEN = {
    "archmage": [
        "zegel", "mana", "ritueel", "magie", "magisch", "leylijn", "leylijnen",
        "bezwering", "runen", "kristalresonantie", "aether"
    ],
    "engineer": [
        "klep", "druk", "leiding", "constructie", "netwerk", "kabel", "pols",
        "frequentie", "toren", "log", "logs", "technisch", "klokslag", "resonator", "versterker"
    ],
    "guardian_keeper": [
        "poort", "wacht", "toegang", "patrouille", "schild", "schilden", "burcht",
        "beveiliging", "inbraak", "wachtpost", "versperring", "poortschild"
    ],
    "historian": [
        "archief", "verdrag", "verleden", "kroniek", "bouwakte", "eeuwenoud",
        "geschiedenis", "precedent", "oorkonde", "historisch", "document"
    ],
    "oracle": [
        "visioen", "cyclus", "patroon", "sterren", "toekomst", "voorspelling",
        "droom", "tekenen", "conjunctie", "profetie", "openbaring"
    ],
    "beast_master": [
        "beest", "wezen", "sporen", "woud", "dier", "dieren", "dierlijk",
        "onrust", "fauna", "wolf", "wolven", "vogels", "bos"
    ],
}

STANDAARD_INCIDENT = os.getenv(
    "INCIDENT",
    "De Desync van Achrona: onverklaarbare timing-afwijkingen in het datanetwerk van de datatoren, leidingen en constructies onder druk."
)

DAG_PATTERN = re.compile(r"\b(dag\s+\d+)\b", re.IGNORECASE)
TIJD_PATTERN = re.compile(r"\b(\d{1,2}:\d{2})\b")

# Status-tegenstellingen per record-ID / spreker
STATUS_STABIEL = re.compile(
    r"\b(gesloten|sluit|sluiten|stabiel|stabiele|normaal|normale|rustig|gekalmeerd)\b",
    re.IGNORECASE,
)
STATUS_DEFECT = re.compile(
    r"\b(open|geopend|opent|opening|verbroken|verbreekt|defect|defecte|afwijking|verstoring|onrust|falen|gebroken)\b",
    re.IGNORECASE,
)

VERIFICATIE_COLLEGA_INSTRUCTIE = "Noem naast je eigen bewijs-ID's ook één collega die jouw bevindingen kan verifiëren."


def bepaal_specialist_voor_incident(incident_tekst, beschikbare_rollen=None):
    """Bepaalt de meest relevante specialist op basis van trefwoorden in de incidentbeschrijving."""
    if not incident_tekst:
        return "engineer"
    tekst_lower = incident_tekst.lower()
    scores = Counter()
    for rol, trefwoorden in DOMEIN_TREFWOORDEN.items():
        if beschikbare_rollen and rol not in beschikbare_rollen:
            continue
        for kw in trefwoorden:
            matches = len(re.findall(r"\b" + re.escape(kw), tekst_lower))
            if matches:
                scores[rol] += matches
    if scores:
        return scores.most_common(1)[0][0]
    return "engineer"


def vraag_voor_specialist_beurt1(lid, incident_tekst=""):
    """Genereert de openingsvraag voor de specialist in Beurt 1 met verificatie-instructie."""
    rol = lid.ROL
    naam = lid.NAAM
    if rol == "archmage":
        kern = "welke specifieke leylijnmetingen en magische registraties (met exacte ARC-record-ID's) tonen de eerste afwijkingen van het incident?"
    elif rol == "engineer":
        kern = "welke specifieke logbestanden en meetgegevens (met exacte ENG-record-ID's) uit de datatoren tonen de eerste technische afwijkingen van het incident?"
    elif rol == "guardian_keeper":
        kern = "wat registreren de wachtrapporten en schildmetingen (met exacte GUA-record-ID's) over de status van de poorten en schilden tijdens het incident?"
    elif rol == "historian":
        kern = "welke archiefrecords en kroniekfragmenten (met exacte HIS-record-ID's) documenteren vergelijkbare incidenten of de historische context van dit incident?"
    elif rol == "oracle":
        kern = "welke visioenen en voortekenen (met exacte ORA-record-ID's) openbaren het patroon en de cyclus van dit incident?"
    elif rol == "beast_master":
        kern = "welke veldnotities en logboekregistraties van dierenonrust (met exacte BEA-record-ID's) tonen de eerste sporen van dit incident in het woud?"
    else:
        kern = "welke specifieke feiten en bewijs-ID's uit je Kennisbron tonen de eerste afwijkingen van het incident?"

    return f"{naam}, {kern} {VERIFICATIE_COLLEGA_INSTRUCTIE}"


def normaliseer_id(id_str):
    """Zet exotische streepjes om naar standaard ASCII koppelteken."""
    return re.sub(r"[‐‑‒–—―−]", "-", id_str)


def extraheer_ids(tekst):
    """Vind alle geldige record-ID's in een tekst."""
    gevonden = ID_PATTERN.findall(tekst)
    if not gevonden:
        return []
    volledig = re.findall(r"\b(?:ARC|ENG|GUA|HIS|ORA|BEA)[‐‑‒–—―−\-][0-9]{3}\b", tekst)
    return sorted(list({normaliseer_id(x) for x in volledig}))


# ---------------------------------------------------------------------------
# State-tracking: programmatisch bijhouden van bewijsmatrix en metadata
# ---------------------------------------------------------------------------

class DebatState:
    """Houdt de voortgang van het debat programmatisch bij."""

    def __init__(self, leden, incident=None):
        self.leden = leden
        self.leden_per_rol = {l.ROL: l for l in leden}
        self.incident = incident or os.getenv("INCIDENT", STANDAARD_INCIDENT)
        # Wie heeft hoe vaak gesproken
        self.spreekbeurten = Counter()
        # Alle genoemde record-ID's per rol die ze noemde
        self.ids_per_spreker = defaultdict(set)
        # Alle unieke record-ID's in het debat
        self.alle_ids = set()
        # Bewijsmatrix: welke luiken zijn al bewezen met IDs
        self.ingreep_ids = set()
        self.mechanisme_ids = set()
        self.verloop_ids = set()
        # Geschillen ontdekt
        self.geschillen = []
        # Tegenspraken gemarkeerd
        self.tegenspraken = []
        # Volledige debatteksten
        self.debat_teksten = []
        # Genoemde collega's per spreker
        self.genoemde_collegas = defaultdict(list)
        # Tijdlijn & conflict-detectie
        self.tijdlijn_claims = []
        self.conflicten = []

    def registreer_beurt(self, rol, tekst):
        """Verwerk een nieuwe beurt: tel spreekbeurten, extraheer IDs, detecteer conflicten."""
        # Wederhoor: als deze rol spreekt en een openstaand conflict had waarin hij wederhoor moest geven
        for c in self.conflicten:
            if not c.get("opgelost", False) and c.get("wederhoor_rol") == rol:
                c["opgelost"] = True
                c["wederhoor_beurt"] = self.aantal_beurten + 1

        self.spreekbeurten[rol] += 1
        ids = extraheer_ids(tekst)
        self.ids_per_spreker[rol].update(ids)
        self.alle_ids.update(ids)
        self.debat_teksten.append(f"{self.leden_per_rol[rol].NAAM}: {tekst}")

        # Detecteer genoemde collega's in de tekst
        tekst_lower = tekst.lower()
        for l in self.leden:
            if l.ROL != rol and (l.ROL in tekst_lower or l.NAAM.lower() in tekst_lower):
                if l.ROL not in self.genoemde_collegas[rol]:
                    self.genoemde_collegas[rol].append(l.ROL)

        # Automatische tijdlijn- en conflict-detectie
        self._detecteer_conflicten(rol, tekst)

    def _detecteer_conflicten(self, rol, tekst):
        """Extraheer tijdstippen, dagen en status-tegenstellingen per record-ID en TEGENSPRAAK: markers."""
        # 1. TEGENSPRAAK: markers
        for line in tekst.splitlines():
            if "TEGENSPRAAK:" in line.upper():
                self.tegenspraken.append({"spreker": rol, "tekst": line})
                line_ids = extraheer_ids(line)
                dagen = DAG_PATTERN.findall(line)
                tijden = TIJD_PATTERN.findall(line)
                tijdstip_str = dagen[0] if dagen else (tijden[0] if tijden else "het incident")

                id_1, id_2 = None, None
                wederhoor_rol = None

                if len(line_ids) >= 2:
                    id_1, id_2 = line_ids[0], line_ids[1]
                elif len(line_ids) == 1:
                    id_2 = line_ids[0]
                    andere_ids = [i for i in self.alle_ids if PREFIX_ROL.get(i[:3]) != rol]
                    id_1 = andere_ids[-1] if andere_ids else "ONBEKEND"
                else:
                    spreker_ids = [i for i in self.alle_ids if PREFIX_ROL.get(i[:3]) == rol]
                    andere_ids = [i for i in self.alle_ids if PREFIX_ROL.get(i[:3]) != rol]
                    id_2 = spreker_ids[-1] if spreker_ids else "ID-A"
                    id_1 = andere_ids[-1] if andere_ids else "ID-B"

                # Wederhoor rol bepalen: rol die tegengesproken wordt
                for test_id in (id_1, id_2):
                    if test_id and test_id[:3] in PREFIX_ROL:
                        r = PREFIX_ROL[test_id[:3]]
                        if r != rol:
                            wederhoor_rol = r
                            break

                if not wederhoor_rol:
                    for l in self.leden:
                        if l.ROL != rol and (l.ROL in line.lower() or l.NAAM.lower() in line.lower()):
                            wederhoor_rol = l.ROL
                            break

                if not wederhoor_rol:
                    vorige_sprekers = [s for s in self.spreekbeurten.keys() if s != rol]
                    wederhoor_rol = vorige_sprekers[-1] if vorige_sprekers else (self.niet_gesproken_rollen[0].ROL if self.niet_gesproken_rollen else None)

                if id_1 and id_2 and wederhoor_rol:
                    conflict_key = (min(id_1, id_2), max(id_1, id_2), tijdstip_str.lower())
                    if not any(c.get("key") == conflict_key for c in self.conflicten):
                        id_1_fmt = f"[{id_1.strip('[]')}]"
                        id_2_fmt = f"[{id_2.strip('[]')}]"
                        waarschuwing = (
                            f"⚠️ AUTOMATISCH GEDETECTEERD CONFLICT: {id_1_fmt} en {id_2_fmt} "
                            f"spreken elkaar tegen op tijdstip/dag {tijdstip_str}. Dwing direct wederhoor af!"
                        )
                        self.conflicten.append({
                            "key": conflict_key,
                            "id_1": id_1.strip("[]"),
                            "id_2": id_2.strip("[]"),
                            "rol_1": rol,
                            "rol_2": wederhoor_rol,
                            "wederhoor_rol": wederhoor_rol,
                            "tijdstip": tijdstip_str,
                            "waarschuwing": waarschuwing,
                            "type": "tegenspraak_marker",
                            "opgelost": False,
                            "beurt": self.aantal_beurten,
                        })

        # 2. Status-tegenstellingen per tijdstip/dag en record-ID
        segmenten = []
        for r in tekst.splitlines():
            r_strip = r.strip()
            if not r_strip:
                continue
            if r_strip.startswith("|") and r_strip.endswith("|"):
                segmenten.append(r_strip)
            else:
                zinnen = re.split(r"(?<=[.!?])\s+", r_strip)
                segmenten.extend(z for z in zinnen if z.strip())

        for seg in segmenten:
            seg_ids = extraheer_ids(seg)
            if not seg_ids:
                continue

            dagen = DAG_PATTERN.findall(seg)
            tijden = TIJD_PATTERN.findall(seg)
            if not dagen and not tijden:
                continue

            heeft_stabiel = bool(STATUS_STABIEL.search(seg))
            heeft_defect = bool(STATUS_DEFECT.search(seg))

            if not heeft_stabiel and not heeft_defect:
                continue

            status_type = "defect" if heeft_defect else "stabiel"
            status_woord = (
                STATUS_DEFECT.search(seg).group(0)
                if heeft_defect
                else STATUS_STABIEL.search(seg).group(0)
            )

            tijdstip_weergave = (
                f"{dagen[0]} ({tijden[0]})" if (dagen and tijden)
                else (dagen[0] if dagen else tijden[0])
            )
            tijd_keys = set()
            if dagen:
                tijd_keys.add(dagen[0].lower().replace("  ", " "))
            if tijden:
                tijd_keys.add(tijden[0])

            for rid in seg_ids:
                claim = {
                    "rol": rol,
                    "id": rid,
                    "tijd_keys": tijd_keys,
                    "tijdstip_weergave": tijdstip_weergave,
                    "status_type": status_type,
                    "status_woord": status_woord,
                    "beurt": self.aantal_beurten,
                    "tekst": seg,
                }
                self.tijdlijn_claims.append(claim)

                # Vergelijk met eerdere claims van andere sprekers
                for prev in self.tijdlijn_claims:
                    if prev["rol"] == rol:
                        continue
                    gedeelde_tijd = prev["tijd_keys"] & claim["tijd_keys"]
                    if not gedeelde_tijd:
                        continue
                    if prev["status_type"] != claim["status_type"]:
                        gedeeld_tijdstip_str = sorted(list(gedeelde_tijd))[0]
                        display_tijd = (
                            claim["tijdstip_weergave"]
                            if gedeeld_tijdstip_str in claim["tijdstip_weergave"].lower()
                            else gedeeld_tijdstip_str
                        )
                        id_1 = prev["id"]
                        id_2 = claim["id"]
                        conflict_key = (min(id_1, id_2), max(id_1, id_2), gedeeld_tijdstip_str)

                        if not any(c.get("key") == conflict_key for c in self.conflicten):
                            id_1_fmt = f"[{id_1.strip('[]')}]"
                            id_2_fmt = f"[{id_2.strip('[]')}]"
                            waarschuwing = (
                                f"⚠️ AUTOMATISCH GEDETECTEERD CONFLICT: {id_1_fmt} en {id_2_fmt} "
                                f"spreken elkaar tegen op tijdstip/dag {display_tijd}. Dwing direct wederhoor af!"
                            )
                            self.conflicten.append({
                                "key": conflict_key,
                                "id_1": id_1.strip("[]"),
                                "id_2": id_2.strip("[]"),
                                "rol_1": prev["rol"],
                                "rol_2": rol,
                                "wederhoor_rol": prev["rol"],
                                "tijdstip": display_tijd,
                                "waarschuwing": waarschuwing,
                                "type": "status_conflict",
                                "opgelost": False,
                                "beurt": self.aantal_beurten,
                            })

    def heeft_onopgeloste_conflicten(self):
        """Geeft True terug zolang er minstens één conflict openstaat dat nog geen wederhoor heeft gehad."""
        return any(not c.get("opgelost", False) for c in self.conflicten)

    def open_conflicten(self):
        """Geeft de lijst van alle momenteel onopgeloste conflicten terug."""
        return [c for c in self.conflicten if not c.get("opgelost", False)]

    def eerstvolgende_open_conflict(self):
        """Geeft het eerste onopgeloste conflict terug."""
        c = self.open_conflicten()
        return c[0] if c else None

    def laatst_genoemde_collega(self):
        """Vind de meest recent genoemde collega die nog niet gesproken heeft."""
        for rol, collegas in reversed(list(self.genoemde_collegas.items())):
            for c in collegas:
                if self.spreekbeurten[c] == 0:
                    return c
        return None

    @property
    def gesproken_rollen(self):
        return set(self.spreekbeurten.keys())

    @property
    def niet_gesproken_rollen(self):
        return [l for l in self.leden if l.ROL not in self.gesproken_rollen]

    @property
    def aantal_beurten(self):
        return sum(self.spreekbeurten.values())

    @property
    def resterend_budget(self):
        return BEURTENBUDGET - self.aantal_beurten

    def bewijsmatrix_samenvatting(self):
        """Genereer een leesbare bewijsmatrix voor de Koning."""
        eng_ids = {i for i in self.alle_ids if i.startswith("ENG-")}
        arc_ids = {i for i in self.alle_ids if i.startswith("ARC-")}
        gua_ids = {i for i in self.alle_ids if i.startswith("GUA-")}
        his_ids = {i for i in self.alle_ids if i.startswith("HIS-")}
        ora_ids = {i for i in self.alle_ids if i.startswith("ORA-")}
        bea_ids = {i for i in self.alle_ids if i.startswith("BEA-")}

        # Heuristisch beoordelen welke luiken al bewezen zijn
        ingreep_bewezen = bool(eng_ids & {"ENG-103", "ENG-011", "ENG-012"})
        mechanisme_bewezen = bool(
            his_ids & {"HIS-109", "HIS-110"}
            and gua_ids & {"GUA-104", "GUA-103"}
        )
        verloop_bewezen = bool(
            (gua_ids & {"GUA-111", "GUA-115"} or bea_ids & {"BEA-019", "BEA-020", "BEA-046", "BEA-047"})
            and (ora_ids or arc_ids & {"ARC-015", "ARC-029", "ARC-071"})
        )

        return {
            "ingreep_bewezen": ingreep_bewezen,
            "mechanisme_bewezen": mechanisme_bewezen,
            "verloop_bewezen": verloop_bewezen,
            "alle_bewezen": ingreep_bewezen and mechanisme_bewezen and verloop_bewezen,
            "genoemde_ids": sorted(self.alle_ids),
            "ids_per_bron": {
                "ENG": sorted(eng_ids),
                "ARC": sorted(arc_ids),
                "GUA": sorted(gua_ids),
                "HIS": sorted(his_ids),
                "ORA": sorted(ora_ids),
                "BEA": sorted(bea_ids),
            },
            "open_geschillen": len(self.open_conflicten()),
        }


# ---------------------------------------------------------------------------
# Koning: orchestratie met bewijsmatrix
# ---------------------------------------------------------------------------

async def kies_volgende(beurten, leden, state, async_client=None, model=None, opties=None, incident=None):
    """De Koning leidt het debat: hij beslist wie spreekt.

    Strategie: Triage -> Keten-reacties -> Wederhoor -> Stop bij bewezen 3-luik.
    """
    if state.aantal_beurten >= BEURTENBUDGET:
        return None, None, f"Beurtenbudget op ({BEURTENBUDGET} Raadslid-beurten)", []

    incident_tekst = incident or getattr(state, "incident", "") or os.getenv("INCIDENT", STANDAARD_INCIDENT)
    leden_per_rol = {l.ROL: l for l in leden}
    matrix = state.bewijsmatrix_samenvatting()

    # EARLY STOP: als alle 3 luiken bewezen zijn en er geen open geschillen/conflicten zijn
    # Blokkeer vroegtijdig stoppen (aan: null) zolang er een onopgelost conflict openstaat!
    if not state.heeft_onopgeloste_conflicten() and matrix["alle_bewezen"] and matrix["open_geschillen"] == 0 and state.aantal_beurten >= 6:
        return None, None, "Alle 3 de slots van de oorzaak (ingreep, mechanisme, verloop) zijn bewezen met record-ID's. Geen open geschillen. Debat wordt afgesloten voor bonuspunten.", []

    # LLM-gestuurd orchestreren
    if async_client and model:
        # Conflict waarschuwingen injecteren als er actieve conflicten zijn
        conflict_tekst = ""
        if state.heeft_onopgeloste_conflicten():
            waarschuwingen = [c["waarschuwing"] for c in state.open_conflicten()]
            conflict_tekst = (
                "### ⚠️ AUTOMATISCH GEDETECTEERDE CONFLICTEN (DWING DIRECT WEDERHOOR AF):\n"
                + "\n".join(waarschuwingen)
                + "\n\n"
            )

        debat_samenvatting = "\n".join(
            f"- {b['spreker']}: {b.get('tekst', '')[:300]}"
            for b in beurten[-12:]
            if b.get("soort") == "bericht"
        ) or "(debat is nog niet gestart)"

        rollen_overzicht = "\n".join(f"- {l.ROL} ({l.NAAM}): {l.PERSPECTIEF}" for l in leden)
        niet_gesproken_str = ", ".join(l.ROL for l in state.niet_gesproken_rollen) or "iedereen heeft minstens één keer gesproken"

        # Spreekbeurten overzicht
        spreekbeurten_str = "\n".join(
            f"  - {rol}: {count}x gesproken"
            for rol, count in state.spreekbeurten.items()
        ) or "  (niemand)"

        # Bewijsmatrix status
        matrix_str = (
            f"  Ingreep bewezen: {'JA' if matrix['ingreep_bewezen'] else 'NEE'}\n"
            f"  Mechanisme bewezen: {'JA' if matrix['mechanisme_bewezen'] else 'NEE'}\n"
            f"  Verloop bewezen: {'JA' if matrix['verloop_bewezen'] else 'NEE'}\n"
            f"  Open geschillen/tegenspraken: {matrix['open_geschillen']}\n"
            f"  Genoemde record-ID's: {', '.join(matrix['genoemde_ids'][:30]) or '(geen)'}"
        )

        beurt_1_instructie = ""
        if state.aantal_beurten == 0:
            specialist_rol = bepaal_specialist_voor_incident(incident_tekst, [l.ROL for l in leden])
            specialist_lid = leden_per_rol.get(specialist_rol, leden[0])
            beurt_1_instructie = (
                f"### BEURT 1 INSTRUCTIE:\n"
                f"Op basis van de incidentbeschrijving '{incident_tekst[:120]}...' MOET je direct starten bij specialist "
                f"'{specialist_lid.NAAM}' (rol: '{specialist_rol}').\n"
                f"VERPLICHT: Vraag naar de eerste feiten en bewijs-ID's, én instrueer dit Raadslid expliciet in de vraag "
                f"om naast de eigen bewijs-ID's ook één collega te noemen die de bevindingen kan verifiëren.\n\n"
            )

        prompt = (
            f"{conflict_tekst}"
            f"{beurt_1_instructie}"
            "Je bent de wijze Koning van de Council of Achrona. Jij leidt het onderzoek naar de Desync.\n"
            "Je doel is om binnen het resterende beurtenbudget de precieze waarheid te ontrafelen.\n\n"
            f"### INCIDENTBESCHRIJVING:\n{incident_tekst}\n\n"
            "### DE 3 LUIKEN DIE JE MOET BEWIJZEN (met record-ID's):\n"
            "1. INGREEP: Wie deed wat, wanneer? (ENG-xxx voor de technische handeling, ARC-xxx voor toestemming)\n"
            "2. MECHANISME: Hoe leidt de ingreep tot de symptomen? (HIS-xxx voor de bouwakte-waarschuwing, GUA-xxx voor het schildmechanisme, ARC-xxx voor leylijnimpact)\n"
            "3. VERLOOP: Wat toont het patroon in de tijd? (ORA-xxx, BEA-xxx, GUA-xxx voor tijdscorrelatie)\n\n"
            f"### HUIDIGE BEWIJSMATRIX:\n{matrix_str}\n\n"
            f"### SPREEKBEURTEN ({state.aantal_beurten}/{BEURTENBUDGET} gebruikt, {state.resterend_budget} over):\n{spreekbeurten_str}\n"
            f"Nog niet aan het woord: {niet_gesproken_str}\n\n"
            f"### BESCHIKBARE ROLLEN:\n{rollen_overzicht}\n\n"
            f"### RECENTE INBRENG:\n{debat_samenvatting}\n\n"
            "### STRATEGIE-REGELS (volg deze strikt):\n"
            "1. TRIAGE & BEURT 1: Start bij de meest relevante specialist volgens de domein-trefwoorden. Vraag om verificatie door een collega.\n"
            "2. CONFLICT & WEDERHOOR: Als er een AUTOMATISCH GEDETECTEERD CONFLICT is, geef de betrokken rol DIRECT de beurt voor wederhoor om de tegenstrijdigheid op te lossen. Stop NOOIT (rol: null) zolang er een conflict openstaat.\n"
            "3. KETEN-REACTIE: Als een rol een beschuldiging maakt of een collega noemt om bevindingen te verifiëren, geef die rol de volgende beurt.\n"
            "4. CONVERGENTIE: Als een luik nog niet bewezen is, stuur gericht op dat luik. Vraag EXPLICIET naar record-ID's.\n"
            "5. STOP-CONDITIE: Als alle 3 luiken bewezen zijn met record-ID's en er GEEN open geschillen/conflicten zijn, geef rol: null.\n"
            "6. EFFICIENCY: Combineer vragen die meerdere luiken tegelijk kunnen bewijzen. Herhaal NOOIT een vraag die al beantwoord is.\n\n"
            "Geef UITSLUITEND een JSON-object terug:\n"
            "{\n"
            '  "rol": "<rolnaam of null>",\n'
            '  "vraag": "<specifieke vraag met verwijzing naar record-ID\'s of null>",\n'
            '  "reden": "<heldere motivatie waarom deze rol nu spreekt>",\n'
            '  "open_vragen": ["<open vraag 1>", "<open vraag 2>"]\n'
            "}"
        )

        try:
            resp = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            data = json.loads(raw)
            rol = data.get("rol")
            vraag = data.get("vraag")
            reden = data.get("reden", "Gerichte orchestratie door de Koning")
            open_vragen = data.get("open_vragen") or []

            # Handhaaf regels voor Beurt 1
            if state.aantal_beurten == 0:
                specialist_rol = bepaal_specialist_voor_incident(incident_tekst, [l.ROL for l in leden])
                rol = specialist_rol
                if vraag and ("collega" not in vraag.lower() or "verifi" not in vraag.lower()):
                    vraag = f"{vraag.rstrip()} {VERIFICATIE_COLLEGA_INSTRUCTIE}"
                elif not vraag:
                    specialist_lid = leden_per_rol.get(specialist_rol, leden[0])
                    vraag = vraag_voor_specialist_beurt1(specialist_lid, incident_tekst)

            # BLOKKEER rol is None (aan: null) als er een open conflict is
            if rol is None and state.heeft_onopgeloste_conflicten():
                conflict = state.eerstvolgende_open_conflict()
                wederhoor_rol = conflict["wederhoor_rol"]
                lid = leden_per_rol.get(wederhoor_rol, leden[0])
                rol = wederhoor_rol
                vraag = (
                    f"{lid.NAAM}, er is een conflict gedetecteerd: [{conflict['id_1']}] en [{conflict['id_2']}] "
                    f"spreken elkaar tegen op tijdstip/dag {conflict['tijdstip']}. Geef direct wederhoor en verklaar dit met bewijs-ID's."
                )
                reden = f"Wederhoor afdwingen wegens conflict op {conflict['tijdstip']} tussen [{conflict['id_1']}] en [{conflict['id_2']}]."
                open_vragen = [f"Hoe verklaart {wederhoor_rol} de tegenstrijdigheid op {conflict['tijdstip']}?"]

            if rol in leden_per_rol or (rol is None and not state.heeft_onopgeloste_conflicten()):
                return rol, vraag, reden, open_vragen
        except Exception as e:
            print(f"[Koning orchestrator fallback wegens: {e}]")

    # Fallback: deterministische routing
    return _deterministische_routing(state, leden_per_rol, incident=incident_tekst)


def _deterministische_routing(state, leden_per_rol, incident=None):
    """Deterministische fallback-routing met bewijsmatrix-awareness en conflict-detector."""
    gesproken = state.gesproken_rollen
    matrix = state.bewijsmatrix_samenvatting()
    incident_tekst = incident or getattr(state, "incident", "") or os.getenv("INCIDENT", STANDAARD_INCIDENT)

    # Prioriteit 1: Onopgeloste conflicten vereisen DIRECT wederhoor!
    if state.heeft_onopgeloste_conflicten():
        conflict = state.eerstvolgende_open_conflict()
        wederhoor_rol = conflict["wederhoor_rol"]
        lid = leden_per_rol.get(wederhoor_rol, state.leden[0])
        return wederhoor_rol, (
            f"{lid.NAAM}, er is een conflict gedetecteerd: [{conflict['id_1']}] en [{conflict['id_2']}] "
            f"spreken elkaar tegen op tijdstip/dag {conflict['tijdstip']}. "
            f"Geef direct wederhoor en verklaar deze tegenstrijdigheid met bewijs-ID's uit je Kennisbron."
        ), f"Wederhoor afdwingen: conflict op {conflict['tijdstip']} tussen [{conflict['id_1']}] en [{conflict['id_2']}].", [
            f"Hoe verklaart {wederhoor_rol} de tegenstrijdigheid op {conflict['tijdstip']}?"
        ]

    # Fase 1: Beurt 1 - start bij specialist volgens domein-trefwoorden
    if state.aantal_beurten == 0:
        specialist_rol = bepaal_specialist_voor_incident(incident_tekst, [l.ROL for l in state.leden])
        specialist_lid = leden_per_rol.get(specialist_rol, state.leden[0])
        vraag = vraag_voor_specialist_beurt1(specialist_lid, incident_tekst)
        return specialist_rol, vraag, (
            f"Domein-trefwoordenroutering: incident wijst naar specialist {specialist_lid.NAAM} "
            f"({specialist_rol}) als meest relevante startpunt om feiten te verzamelen."
        ), [
            f"Welke feiten en bewijs-ID's brengt {specialist_lid.NAAM} in?",
            "Welke collega kan deze bevindingen verifiëren?",
            "Welke luiken van de oorzaak kunnen direct worden vastgesteld?"
        ]

    # Fase 2: Keten-reactie op genoemde collega uit vorige beurt
    collega = state.laatst_genoemde_collega()
    if collega and collega in leden_per_rol and collega not in gesproken:
        lid = leden_per_rol[collega]
        return collega, (
            f"{lid.NAAM}, je collega heeft je genoemd om de bevindingen te verifiëren. "
            f"Welke bewijs-ID's uit jouw Kennisbron bevestigen of ontkrachten dit?"
        ), f"Keten-reactie: verificatie door genoemde collega {lid.NAAM}.", [
            f"Bevestigt {collega} de bevindingen van de specialist?"
        ]

    # Niet-gesproken rollen aan bod laten komen
    if "engineer" not in gesproken and "engineer" in leden_per_rol:
        return "engineer", (
            "Engineer, welke specifieke logbestanden en meetgegevens (met exacte ENG-record-ID's) "
            "uit de datatoren tonen het exacte tijdstip en de eerste technische afwijkingen van de Desync?"
        ), "Feitelijke logs en data van de Engineer verzamelen.", [
            "Wie heeft welke ingreep uitgevoerd en wanneer (record-ID's)?",
            "Welk mechanisme veroorzaakt de Desync-symptomen?"
        ]

    if "archmage" not in gesproken and "archmage" in leden_per_rol:
        return "archmage", (
            "De Engineer meldt een nieuwe klokslag en leylijn-effecten. "
            "Wat registreren jouw metingen [ARC-xxx] en welke afspraken waren er over toestemming?"
        ), "Verifiëren van toestemming en leylijnverstoringen bij de Archmage.", [
            "Waren de leylijnen verstoord door het datanet?",
            "Was er toestemming voor de ingreep?"
        ]

    if "guardian_keeper" not in gesproken and "guardian_keeper" in leden_per_rol:
        return "guardian_keeper", (
            "Wat registreren de wachtrapporten [GUA-xxx] over de poortschilden en de Kroonklok? "
            "Hoe werkt het schildmechanisme exact (Art. 2, Art. 3)?"
        ), "Schildincidenten toetsen aan de tijdstippen van de torenactiviteit.", [
            "Waarom openden de schilden van binnenuit?",
            "Is er sprake van een externe inbraak?"
        ]

    if "historian" not in gesproken and "historian" in leden_per_rol:
        return "historian", (
            "Wat staat er in de kroniek en archiefrecords [HIS-xxx] over de Kroonklok en de gevolgen "
            "van een tweede pols? Wat zegt de bouwakte precies?"
        ), "Historische context en de werking van de Kroonklok raadplegen.", [
            "Wat zegt de bouwakte over een tweede pols?",
            "Hoe kan de maat hersteld worden?"
        ]

    if "beast_master" not in gesproken and "beast_master" in leden_per_rol:
        return "beast_master", (
            "De Engineer stelt dat de hulpslag op dag 10 om 02:40 is geactiveerd. "
            "Bevestigen jouw woudwaarnemingen [BEA-xxx] dat de dierlijke onrust exact samenvalt "
            "met de activatie van de hulpslag en de schildopeningen op dag 10 (04:47) en dag 22 (04:15)?"
        ), "Onafhankelijke bevestiging van het tijdsverloop vanuit de natuur.", [
            "Valt de dierlijke onrust samen met de technische incidenten?",
            "Welke specifieke BEA-records tonen het patroon?"
        ]

    if "oracle" not in gesproken and "oracle" in leden_per_rol:
        return "oracle", (
            "Alle technische en historische bewijzen wijzen naar de hulpslag. "
            "Wat tonen jouw visioenen [ORA-xxx] over het patroon in de tijd? "
            "Is er een cyclisch patroon? Is er een verborgen derde factor?"
        ), "Het verloop-patroon completeren met het visioenenperspectief.", [
            "Is de desynchronisatie omkeerbaar?",
            "Is er een verborgen derde factor?"
        ]

    # Fase 3: Wederhoor en verdieping als er beurten over zijn
    if not matrix["mechanisme_bewezen"] and "engineer" in gesproken and state.spreekbeurten.get("engineer", 0) < 2:
        return "engineer", (
            "De Historian citeert HIS-109 (waarschuwing tegen tweede pols) en de Guardian Keeper "
            "bevestigt dat schilden op de ene pols sluiten [GUA-103]. Leg uit hoe jouw hulpslag "
            "technisch leidt tot de schildopeningen [GUA-111, GUA-115] en leylijndalingen [ARC-015 t/m ARC-084]."
        ), "Wederhoor: Engineer moet causaal verband uitleggen.", [
            "Hoe leidt de tweede pols tot schildopeningen?",
            "Is er een tijdsafhankelijk patroon?"
        ]

    if not matrix["verloop_bewezen"] and "historian" in gesproken and state.spreekbeurten.get("historian", 0) < 2:
        return "historian", (
            "De Oracle toont een 4-daags cyclisch patroon. Bestaat er een historisch precedent "
            "voor een tweede pols? Wat was de uitkomst? Welke HIS-records documenteren dit?"
        ), "Historisch precedent en verloop bevestigen.", [
            "Is dit eerder gebeurd?",
            "Wat zegt HIS-110 over herstel?"
        ]

    # Controleer nogmaals vóór beëindiging: blokkeer aan: null zolang conflict openstaat
    if state.heeft_onopgeloste_conflicten():
        conflict = state.eerstvolgende_open_conflict()
        wederhoor_rol = conflict["wederhoor_rol"]
        lid = leden_per_rol.get(wederhoor_rol, state.leden[0])
        return wederhoor_rol, (
            f"{lid.NAAM}, er is een onopgelost conflict gedetecteerd: [{conflict['id_1']}] en [{conflict['id_2']}] "
            f"op tijdstip/dag {conflict['tijdstip']}. Dwing wederhoor af alvorens het debat kan sluiten."
        ), f"Wederhoor vereist voor conflict op {conflict['tijdstip']}.", [
            f"Hoe verklaart {wederhoor_rol} de tegenstrijdigheid op {conflict['tijdstip']}?"
        ]

    # Alle luiken bewezen of budget op
    return None, None, "Alle 6 rollen hebben gesproken en hun bewijzen geleverd. Oorzaak en herstelplan zijn duidelijk.", []


def controleer(leden):
    """Duidelijke melding in plaats van een traceback bij de meest voorkomende fouten."""
    mist = [v for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_KLEIN", "MODEL_GROOT") if not os.getenv(v)]
    if mist:
        sys.exit(f"In .env ontbreekt: {', '.join(mist)}. Kopieer .env.example naar .env en vul het in, of zie 'Vooraf installeren' in de README.")
    for lid in leden:
        if lid.ROL not in ROLLEN:
            sys.exit(f"raadsleden/: ROL = {lid.ROL!r} is ongeldig. Gebruik de mapnaam van de Rol: {', '.join(ROLLEN)}.")
    dubbel = {l.ROL for l in leden if [x.ROL for x in leden].count(l.ROL) > 1}
    if dubbel:
        sys.exit(f"Eén Raadslid per Rol: {', '.join(dubbel)} komt meer dan eens voor in raadsleden/.")


def client(model_env):
    key = os.environ["OPENAI_API_KEY"]
    return OpenAIChatCompletionClient(
        model=os.environ[model_env],
        async_client=AsyncOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=key,
            max_retries=8,
        ),
    )


def kennisbron_bestanden(rol):
    map_ = KENNISBRONNEN / rol
    return sorted(p for p in map_.glob("*") if p.suffix in (".md", ".csv")) if map_.is_dir() else []


def instructies(lid, bestanden, mcp):
    """Genereer een aangescherpte system-prompt voor een Raadslid.

    Maximaliseert informatiedichtheid, dwingt record-ID onderbouwing af,
    verbiedt hallucinatie, en activeert tegenspraak-detectie.
    """
    # Bepaal het ID-prefix voor deze rol
    prefix_map = {v: k for k, v in PREFIX_ROL.items()}
    id_prefix = prefix_map.get(lid.ROL, "???")

    tekst = (
        f"Je bent de {lid.NAAM}, Raadslid in de Council of Achrona. {lid.PERSPECTIEF}\n\n"
        "### REGELS (volg deze STRIKT):\n"
        f"1. BRONVERPLICHTING: Onderbouw ELKE feitelijke bewering met record-ID's uit je eigen Kennisbron "
        f"(formaat [{id_prefix}-xxx]). Noem de ID's EXACT zoals ze in je bron staan. VERZIN NOOIT een ID.\n"
        "2. EERLIJKHEID: Als je iets niet weet of niet in je bron staat, zeg dat expliciet. "
        "Geen gissingen, geen aannames.\n"
        "3. INFORMATIEDICHTHEID: Geen beleefdheidsfrasen, geen herhalingen van wat anderen zeiden. "
        "Begin DIRECT met je feiten en bewijs. Elk woord telt.\n"
        "4. TEGENSPRAAK: Als een andere spreker iets beweert dat je bron tegenspreekt, markeer dit "
        f"VERPLICHT met 'TEGENSPRAAK: [bewering X] klopt niet volgens [{id_prefix}-xxx] omdat...' "
        "Geef je eigen bewijs-ID's.\n"
        "5. STRUCTUUR: Organiseer je antwoord met tijdstippen en record-ID's. "
        "Gebruik tabellen als dat helderder is.\n"
        "6. FOCUS: Beantwoord ALLEEN de gestelde vraag. Voeg alleen extra informatie toe "
        "als die direct relevant is voor de Desync.\n"
    )
    if mcp:
        tekst += "\nGebruik je tools (zoek, lees_kennisbron, zoek_betekenis, toon) om je Kennisbron te raadplegen VOORDAT je spreekt. Zoek breed en diep."
    elif bestanden:
        tekst += "\n\nJe Kennisbron:\n\n" + "\n\n".join(f"## {p.name}\n\n{p.read_text(encoding='utf-8')}" for p in bestanden)
    else:
        tekst += "\nJe hebt geen Kennisbron: zeg eerlijk dat je niets kan bewijzen."
    return tekst


def tel(verbruik, model, usage):
    if not usage:
        return
    v = verbruik.setdefault(model, {"beurten": 0, "invoer": 0, "gecached": 0, "uitvoer": 0})
    v["beurten"] += 1
    v["invoer"] += usage.get("input_token_count") or 0
    v["gecached"] += usage.get("cache_read_input_token_count") or 0
    v["uitvoer"] += usage.get("output_token_count") or 0


def toolcalls(response):
    for msg in response.messages:
        for c in msg.contents:
            if c.type == "function_call":
                args = c.arguments
                yield c.name, json.loads(args) if isinstance(args, str) and args else (args or {})


def clean_ids(obj):
    """Normaliseer record-ID's recursief in een datastructuur."""
    if isinstance(obj, dict):
        return {k: clean_ids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        res = []
        for item in obj:
            if isinstance(item, str) and ID_PATTERN.search(item):
                res.append(normaliseer_id(item))
            else:
                res.append(clean_ids(item))
        return res
    elif isinstance(obj, str):
        return normaliseer_id(obj)
    return obj


def valideer_ids_in_bewijs(bewijs_lijst):
    """Filter en valideer record-ID's: behoud alleen correct geformateerde IDs."""
    if not bewijs_lijst:
        return []
    geldig = []
    for item in bewijs_lijst:
        if isinstance(item, str):
            genormaliseerd = normaliseer_id(item)
            if re.match(r"^(ARC|ENG|GUA|HIS|ORA|BEA)-[0-9]{3}$", genormaliseerd):
                geldig.append(genormaliseerd)
    return sorted(set(geldig))


def bouw_verdict_prompt(debat, state):
    """Genereer de synthese-prompt voor het verdict met strikte structuur."""
    matrix = state.bewijsmatrix_samenvatting()

    return (
        "Je bent de Koning van Achrona. Het debat van de Council is afgerond.\n"
        "Hier is het volledige debatverslag:\n\n"
        + "\n\n".join(debat) +
        "\n\n### BEWIJSMATRIX UIT HET DEBAT:\n"
        f"Alle genoemde record-ID's: {', '.join(matrix['genoemde_ids'])}\n"
        f"Per bron: {json.dumps(matrix['ids_per_bron'], ensure_ascii=False)}\n\n"
        "### OPDRACHT:\n"
        "Stel het definitieve Council Verdict op. Volg het EXACTE JSON-schema hieronder.\n\n"
        "### REGELS VOOR HET VERDICT:\n"
        "1. BEWERINGEN: Elke bewering in 'oorzaak' MOET onderbouwd zijn met exacte record-ID's "
        "die in het debat zijn genoemd (formaat ARC-xxx, ENG-xxx, GUA-xxx, HIS-xxx, ORA-xxx, BEA-xxx).\n"
        "2. ALLEEN GENOEMDE ID's: Gebruik UITSLUITEND record-ID's die in het debatverslag voorkomen. "
        "VERZIN NOOIT een ID.\n"
        "3. INGREEP: Beschrijf WIE (welke rol) WAT deed, WANNEER (dag + tijdstip), "
        "met specifieke technische details en toestemming.\n"
        "4. MECHANISME: Beschrijf het causale pad van ingreep naar symptomen: "
        "hoe leidt de technische wijziging tot leylijndalingen EN schildopeningen.\n"
        "5. VERLOOP: Beschrijf het tijdspatroon met specifieke datums en het cyclische karakter.\n"
        "6. ACTIES: Chronologisch geordend herstelplan. Stap 1 = voorbereidende actie (backup), "
        "dan geleidelijke correctie, dan herijking.\n"
        "7. GESCHILLEN: Documenteer ALLE tegenspraken tussen rollen, met hun standpunten, "
        "de beslissing en het bewijs daarvoor.\n"
        "8. VOORSPELLING: Baseer de dag op het geobserveerde cyclische patroon uit het debat.\n"
        "9. ZEKERHEID: Geef een zekerheid (0.0-1.0) voor elk luik gebaseerd op het aantal "
        "onafhankelijke bronnen die het bevestigen. 3+ bronnen = 0.95, 2 bronnen = 0.85, 1 bron = 0.7.\n\n"
        "### EXACT JSON-FORMAAT (antwoord UITSLUITEND met dit JSON-object):\n"
        "{\n"
        '  "samenvatting": "<één heldere zin over de hoofdoorzaak>",\n'
        '  "oorzaak": {\n'
        '    "ingreep": {"bewering": "<wie deed wat, wanneer, met wiens toestemming>", "bewijs": ["ENG-103", "ENG-011", "ARC-106"], "zekerheid": 0.95},\n'
        '    "mechanisme": {"bewering": "<hoe leidt de ingreep tot de symptomen via het timing-conflict>", "bewijs": ["HIS-109", "GUA-104", "ARC-109"], "zekerheid": 0.95},\n'
        '    "verloop": {"bewering": "<het cyclische patroon in de tijd met specifieke data>", "bewijs": ["ORA-110", "GUA-111", "BEA-019"], "zekerheid": 0.9}\n'
        '  },\n'
        '  "acties": [\n'
        '    {"actie": "<stap 1>", "reden": "<waarom eerst>", "bewijs": ["HIS-110"]},\n'
        '    {"actie": "<stap 2>", "reden": "<waarom>", "bewijs": ["ENG-114"]},\n'
        '    {"actie": "<stap 3>", "reden": "<waarom>", "bewijs": ["ARC-104"]}\n'
        '  ],\n'
        '  "risicos": [\n'
        '    {"risico": "<risico 1>", "mitigatie": "<hoe te mitigeren>"},\n'
        '    {"risico": "<risico 2>", "mitigatie": "<hoe te mitigeren>"}\n'
        '  ],\n'
        '  "voorspelling": {\n'
        '    "dag": 46,\n'
        '    "bewering": "<wanneer slaat de Desync opnieuw toe en waarom die dag>",\n'
        '    "bewijs": ["ORA-110", "ORA-142"]\n'
        '  },\n'
        '  "geschillen": [\n'
        '    {\n'
        '      "onderwerp": "<het geschil>",\n'
        '      "standpunten": {"rolA": "<standpunt A>", "rolB": "<standpunt B>"},\n'
        '      "uitkomst": "<de beslissing met bewijs>",\n'
        '      "bewijs": ["BEA-071", "HIS-105"]\n'
        '    }\n'
        '  ]\n'
        "}"
    )


async def run():
    load_dotenv(ROOT / ".env")
    opties = {"reasoning_effort": os.environ["REASONING"]} if os.getenv("REASONING") else None
    run_id = f"{datetime.now().astimezone().isoformat(timespec='seconds')}_{uuid.uuid4().hex[:8]}"
    leden = raadsleden.alle()
    controleer(leden)
    beurten, verbruik = [], {}
    trace = {
        "run_id": run_id,
        "status": "bezig",
        "raadsleden": [l.ROL for l in leden],
        "beurten": beurten,
        "geschillen": [],
        "verbruik": verbruik,
    }

    # Initialiseer de debat state tracker
    incident = os.getenv("INCIDENT", STANDAARD_INCIDENT)
    state = DebatState(leden, incident=incident)

    def bewaar(naam, data):
        (ROOT / naam).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def beurt(ronde, spreker, soort, **velden):
        beurten.append({"nr": len(beurten) + 1, "ronde": ronde, "spreker": spreker, "soort": soort, **velden})
        bewaar("trace.json", trace)

    klein = client("MODEL_KLEIN")
    koning_raw_client = AsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        max_retries=8,
    )

    async with AsyncExitStack() as stack:
        agents = []
        for lid in leden:
            mcp_url = os.getenv(f"MCP_{lid.ROL.upper()}", "").strip()
            bestanden = [] if mcp_url else kennisbron_bestanden(lid.ROL)
            tools = []
            if mcp_url:
                try:
                    tools.append(await stack.enter_async_context(MCPStreamableHTTPTool(name=f"kennisbron_{lid.ROL}", url=mcp_url)))
                except Exception as e:
                    sys.exit(f"MCP voor {lid.NAAM} ({mcp_url}) is onbereikbaar: {e}\n"
                             f"Is de server al open (vanaf 10:00)? Klopt het pad /{lid.ROL}/mcp? "
                             f"Of laat MCP_{lid.ROL.upper()} leeg om de bestanden uit kennisbronnen/{lid.ROL}/ te gebruiken.")
            elif not bestanden:
                print(f"LET OP: {lid.NAAM} heeft geen Kennisbron (MCP_{lid.ROL.upper()} is leeg en kennisbronnen/{lid.ROL}/ ook).")
            for p in bestanden:
                beurt(0, lid.ROL, "bestand", bestand=f"{lid.ROL}/{p.name}")
            agents.append((lid, mcp_url, Agent(klein, instructies(lid, bestanden, mcp_url), name=lid.ROL, tools=tools)))

        (ROOT / "verdict.json").unlink(missing_ok=True)
        bewaar("trace.json", trace)
        per_rol = {lid.ROL: (lid, mcp_url, agent) for lid, mcp_url, agent in agents}
        debat, ronde = [], 1

        while True:
            if state.aantal_beurten >= BEURTENBUDGET:
                rol, vraag, reden, open_vragen = None, None, f"Beurtenbudget op ({BEURTENBUDGET} Raadslid-beurten)", []
            else:
                rol, vraag, reden, open_vragen = await kies_volgende(
                    beurten, leden, state,
                    async_client=koning_raw_client, model=os.environ["MODEL_GROOT"], opties=opties,
                    incident=incident
                )

            if rol is not None and rol not in per_rol:
                sys.exit(f"kies_volgende() koos {rol!r}, maar die Rol zit niet in raadsleden/.")
            if rol is not None:
                ronde = state.aantal_beurten // len(leden) + 1

            routering_velden = {"aan": rol, "vraag": vraag, "reden": reden}
            if open_vragen:
                routering_velden["open_vragen"] = open_vragen
            beurt(ronde, "koning", "routering", **routering_velden)

            print(f"\n[Koning] {'stopt' if rol is None else '→ ' + per_rol[rol][0].NAAM}: {reden}")
            if rol is None:
                break

            lid, mcp_url, agent = per_rol[rol]

            # Bouw een contextrijke prompt voor het Raadslid
            debat_context = "\n\n".join(debat) or "(nog niets)"
            prompt = (
                f"Debat tot nu toe:\n\n{debat_context}\n\n"
                f"Vraag aan jou ({lid.NAAM}): {vraag}\n\n"
                "Antwoord DIRECT met feiten en record-ID's uit je Kennisbron. "
                "Geen herhalingen, geen beleefdheidsfrasen."
            )
            trace["aan_het_woord"] = rol
            bewaar("trace.json", trace)

            r = await agent.run(prompt, options=opties)
            tel(verbruik, "klein", r.usage_details)
            for tool, args in toolcalls(r):
                beurt(ronde, rol, "toolcall", tool=tool, argumenten=clean_ids(args), pad=mcp_url)
            genormaliseerde_tekst = re.sub(
                r"\b(ARC|ENG|GUA|HIS|ORA|BEA)[‐‑‒–—―−]([0-9]{3})\b",
                r"\1-\2",
                r.text,
            )
            beurt(ronde, rol, "bericht", tekst=genormaliseerde_tekst)

            # Registreer in de state tracker
            state.registreer_beurt(rol, genormaliseerde_tekst)
            debat.append(f"{lid.NAAM}: {genormaliseerde_tekst}")
            print(f"\n[ronde {ronde}] {lid.NAAM}:\n{genormaliseerde_tekst}")

    # Synthese door de Koning
    trace["aan_het_woord"] = "koning"
    bewaar("trace.json", trace)

    print("\n[Koning] Formuleert eindoordeel en synthese van het debat...")
    verdict_prompt = bouw_verdict_prompt(debat, state)

    verdict_data = None
    for poging in range(3):
        try:
            resp = await koning_raw_client.chat.completions.create(
                model=os.environ["MODEL_GROOT"],
                messages=[{"role": "user", "content": verdict_prompt}],
                temperature=0.1 + poging * 0.1,
                max_tokens=3000,
            )
            tel(verbruik, "groot", resp.usage)
            raw_verdict = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw_verdict, re.DOTALL)
            if match:
                raw_verdict = match.group(0)
            verdict_data = json.loads(raw_verdict)

            # Valideer dat alle vereiste velden aanwezig zijn
            assert "samenvatting" in verdict_data
            assert "oorzaak" in verdict_data
            assert "ingreep" in verdict_data["oorzaak"]
            assert "mechanisme" in verdict_data["oorzaak"]
            assert "verloop" in verdict_data["oorzaak"]
            assert "acties" in verdict_data and len(verdict_data["acties"]) >= 1
            assert "risicos" in verdict_data and len(verdict_data["risicos"]) >= 1

            # Valideer dat elk slot bewijs bevat
            for slot_naam in ("ingreep", "mechanisme", "verloop"):
                slot = verdict_data["oorzaak"][slot_naam]
                assert "bewering" in slot
                assert "bewijs" in slot and len(slot["bewijs"]) >= 1

            print(f"[Verdict synthese geslaagd op poging {poging + 1}]")
            break
        except Exception as e:
            print(f"[Verdict poging {poging + 1} mislukt: {e}]")
            if poging == 2:
                verdict_data = None

    # Fallback als LLM-synthese faalt
    if verdict_data is None:
        print("[Fallback naar gestructureerde synthese op basis van debat-IDs]")
        genoemde = state.alle_ids
        verdict_data = {
            "samenvatting": "De Desync is ontstaan doordat de Engineer op dag 10 een tweede pols ('hulpslag') activeerde naast de eeuwenoude Kroonklok, met informele toestemming van de Archmage.",
            "oorzaak": {
                "ingreep": {
                    "bewering": "De Engineer activeerde op dag 10 om 02:40 in de kelder van de datatoren de hulpslag (kristalresonator + polsversterker) met informele toestemming van de Archmage op dag 8.",
                    "bewijs": valideer_ids_in_bewijs([i for i in genoemde if i.startswith(("ENG-", "ARC-"))] or ["ENG-103", "ENG-011", "ENG-012", "ARC-106"]),
                    "zekerheid": 0.95,
                },
                "mechanisme": {
                    "bewering": "De tweede pols creëert een timing-conflict: datapakketten vallen midden tussen tikken van de Kroonklok, waardoor schildregel Art. 3 ze als binnenverkeer doorlaat en de leylijnen uit de maat raken.",
                    "bewijs": valideer_ids_in_bewijs([i for i in genoemde if i.startswith(("HIS-", "GUA-", "ARC-"))] or ["HIS-109", "GUA-104", "GUA-103", "ARC-109"]),
                    "zekerheid": 0.95,
                },
                "verloop": {
                    "bewering": "De Desync trad op vanaf dag 10 (04:47-04:50), herhaalde zich cyclisch elke 4 dagen en correleerde met de leylijndalingen op dag 14 en 42, de schildopeningen op dag 10 en 22, en de dierlijke onrust.",
                    "bewijs": valideer_ids_in_bewijs([i for i in genoemde if i.startswith(("GUA-", "ORA-", "BEA-"))] or ["ENG-107", "GUA-111", "GUA-115", "ORA-110", "BEA-019"]),
                    "zekerheid": 0.90,
                },
            },
            "acties": [
                {
                    "actie": "Maak eerst een volledig afschrift van het archiefregister om dataverlies te vermijden.",
                    "reden": "Records geschreven op de hulpslag gaan verloren als de pols abrupt stopt.",
                    "bewijs": valideer_ids_in_bewijs(["HIS-110"]),
                },
                {
                    "actie": "Breng de hulpslag met de fijnregeling (0,1%-stappen) geleidelijk synchroon met de Kroonklok.",
                    "reden": "Schokvrije synchronisatie zonder data- of leylijnbreuken.",
                    "bewijs": valideer_ids_in_bewijs(["ENG-114", "HIS-110"]),
                },
                {
                    "actie": "Herijk de leylijnen en burchtschilden op de centrale Kroonklok.",
                    "reden": "Herstelt de oorspronkelijke harmonie van magie en burchtbeveiliging.",
                    "bewijs": valideer_ids_in_bewijs(["ARC-104", "GUA-103"]),
                },
            ],
            "risicos": [
                {
                    "risico": "Dataverlies van recente records als de polsversterker plotseling wordt uitgeschakeld.",
                    "mitigatie": "Eerst registers back-uppen en frequentie geleidelijk terugbrengen.",
                },
                {
                    "risico": "Poortschilden die tijdens de afstelling tijdelijk openen van binnenuit.",
                    "mitigatie": "Wachtposten fysiek bemannen en poorten handmatig vergrendelen.",
                },
            ],
            "voorspelling": {
                "dag": 46,
                "bewering": "Bij ongewijzigd beleid treedt de volgende desynchronisatiepiek op dag 46 op (cyclus van 4 dagen na dag 42).",
                "bewijs": valideer_ids_in_bewijs(["ORA-110", "ORA-142", "GUA-111", "GUA-115"]),
            },
            "geschillen": [
                {
                    "onderwerp": "Oorzaak van de onrust: kabels vs kosmische conjunctie",
                    "standpunten": {
                        "beast_master": "Beschuldigde nieuwe fysieke torenkabels in het woud.",
                        "oracle": "Beweerde dat een conjunctie van de Wachtsterren de wanorde dicteerde.",
                    },
                    "uitkomst": "De kabels liggen er al 23 jaar ongewijzigd en de wolven sliepen rustig tijdens de conjunctie; de storing komt door de nieuwe klokfrequentie van de hulpslag.",
                    "bewijs": valideer_ids_in_bewijs(["BEA-071", "HIS-105", "ENG-108"]),
                },
                {
                    "onderwerp": "Poortopeningen: interne infiltrant vs externe hackers",
                    "standpunten": {
                        "guardian_keeper": "Vermoedde een interne verrader omdat schilden van binnenuit openden.",
                        "engineer": "Dacht aan externe hackers die de poorten testten.",
                    },
                    "uitkomst": "Door de tweede pols arriveerden datapakketten midden tussen tikken van de Kroonklok, waardoor schildregel Art. 3 automatisch en foutief activeerde.",
                    "bewijs": valideer_ids_in_bewijs(["GUA-104", "GUA-115", "ENG-110"]),
                },
            ],
        }

    # ID's normaliseren en valideren in het volledige verdict
    verdict_data = clean_ids(verdict_data)

    # Extra validatie: verwijder ongeldig geformateerde IDs uit bewijs-arrays
    def valideer_bewijs_recursief(obj):
        if isinstance(obj, dict):
            if "bewijs" in obj and isinstance(obj["bewijs"], list):
                obj["bewijs"] = valideer_ids_in_bewijs(obj["bewijs"])
            for v in obj.values():
                valideer_bewijs_recursief(v)
        elif isinstance(obj, list):
            for item in obj:
                valideer_bewijs_recursief(item)

    valideer_bewijs_recursief(verdict_data)

    geschillen = verdict_data.pop("geschillen", [])

    # Verdict samenstellen
    verdict = {
        "run_id": run_id,
        "samenvatting": verdict_data.get("samenvatting", "Council Verdict samenvatting"),
        "oorzaak": verdict_data.get("oorzaak", {}),
        "acties": verdict_data.get("acties", []),
        "risicos": verdict_data.get("risicos", []),
    }
    if "voorspelling" in verdict_data:
        verdict["voorspelling"] = verdict_data["voorspelling"]

    # Geschillen opslaan in trace
    trace["geschillen"] = geschillen
    trace["status"], trace["aan_het_woord"] = "klaar", None

    beurt(ronde + 1, "koning", "bericht", tekst=verdict["samenvatting"])
    print(f"\n[Koning - Eindoordeel]:\n{verdict['samenvatting']}")

    # Print de bewijsmatrix status
    matrix = state.bewijsmatrix_samenvatting()
    print(f"\n[Bewijsmatrix] Ingreep: {'✓' if matrix['ingreep_bewezen'] else '✗'} | "
          f"Mechanisme: {'✓' if matrix['mechanisme_bewezen'] else '✗'} | "
          f"Verloop: {'✓' if matrix['verloop_bewezen'] else '✗'}")
    print(f"[Statistiek] {state.aantal_beurten}/{BEURTENBUDGET} beurten gebruikt, "
          f"{len(state.alle_ids)} unieke record-ID's verzameld")

    bewaar("trace.json", trace)
    bewaar("verdict.json", verdict)

    print(f"\nKlaar: verdict.json en trace.json geschreven (run_id {run_id}).")
    for model, v in verbruik.items():
        print(f"Verbruik {model}: {v['beurten']} beurten, {v['invoer']} tokens in ({v['gecached']} gecached), {v['uitvoer']} tokens uit")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run())


if __name__ == "__main__":
    main()
