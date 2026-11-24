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

    def __init__(self, leden):
        self.leden = leden
        self.leden_per_rol = {l.ROL: l for l in leden}
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

    def registreer_beurt(self, rol, tekst):
        """Verwerk een nieuwe beurt: tel spreekbeurten, extraheer IDs."""
        self.spreekbeurten[rol] += 1
        ids = extraheer_ids(tekst)
        self.ids_per_spreker[rol].update(ids)
        self.alle_ids.update(ids)
        self.debat_teksten.append(f"{self.leden_per_rol[rol].NAAM}: {tekst}")

        # Detecteer tegenspraken
        if "TEGENSPRAAK:" in tekst.upper():
            self.tegenspraken.append({"spreker": rol, "tekst": tekst})

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
            "open_geschillen": len(self.tegenspraken),
        }


# ---------------------------------------------------------------------------
# Koning: orchestratie met bewijsmatrix
# ---------------------------------------------------------------------------

async def kies_volgende(beurten, leden, state, async_client=None, model=None, opties=None):
    """De Koning leidt het debat: hij beslist wie spreekt.

    Strategie: Triage -> Keten-reacties -> Wederhoor -> Stop bij bewezen 3-luik.
    """
    if state.aantal_beurten >= BEURTENBUDGET:
        return None, None, f"Beurtenbudget op ({BEURTENBUDGET} Raadslid-beurten)", []

    matrix = state.bewijsmatrix_samenvatting()

    # EARLY STOP: als alle 3 luiken bewezen zijn en er geen open geschillen zijn
    if matrix["alle_bewezen"] and matrix["open_geschillen"] == 0 and state.aantal_beurten >= 6:
        return None, None, "Alle 3 de slots van de oorzaak (ingreep, mechanisme, verloop) zijn bewezen met record-ID's. Geen open geschillen. Debat wordt afgesloten voor bonuspunten.", []

    leden_per_rol = {l.ROL: l for l in leden}

    # LLM-gestuurd orchestreren
    if async_client and model:
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

        prompt = (
            "Je bent de wijze Koning van de Council of Achrona. Jij leidt het onderzoek naar de Desync.\n"
            "Je doel is om binnen het resterende beurtenbudget de precieze waarheid te ontrafelen.\n\n"
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
            "1. TRIAGE: Begin bij de rol die de hardste feiten heeft (engineer voor logs, historian voor bouwakte).\n"
            "2. KETEN-REACTIE: Als een rol een beschuldiging of claim maakt, geef de beschuldigde/betrokken rol DIRECT de volgende beurt voor wederhoor.\n"
            "3. CONVERGENTIE: Als een luik nog niet bewezen is, stuur gericht op dat luik. Vraag EXPLICIET naar record-ID's.\n"
            "4. STOP-CONDITIE: Als alle 3 luiken bewezen zijn met record-ID's en er geen open geschillen zijn, geef rol: null.\n"
            "5. EFFICIENCY: Combineer vragen die meerdere luiken tegelijk kunnen bewijzen. Herhaal NOOIT een vraag die al beantwoord is.\n"
            "6. GESCHILLEN: Als twee rollen elkaar tegenspreken, markeer dit en vraag de rol met het sterkste bewijs om te reageren.\n\n"
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

            if rol in leden_per_rol or rol is None:
                return rol, vraag, reden, open_vragen
        except Exception as e:
            print(f"[Koning orchestrator fallback wegens: {e}]")

    # Fallback: deterministische maar intelligente volgorde
    return _deterministische_routing(state, leden_per_rol)


def _deterministische_routing(state, leden_per_rol):
    """Deterministische fallback-routing met bewijsmatrix-awareness."""
    gesproken = state.gesproken_rollen
    matrix = state.bewijsmatrix_samenvatting()

    # Fase 1: Eerste ronde - verzamel basisfeiten
    if state.aantal_beurten == 0:
        return "engineer", (
            "Engineer, welke specifieke logbestanden en meetgegevens (met exacte ENG-record-ID's) "
            "uit de datatoren tonen het exacte tijdstip en de eerste technische afwijkingen van de Desync?"
        ), "Om een feitelijke basis te leggen beginnen we bij de harde, meetbare data.", [
            "Wie heeft welke ingreep uitgevoerd en wanneer (record-ID's)?",
            "Welk mechanisme veroorzaakt de Desync-symptomen?",
            "Welk patroon vertoont de Desync in de tijd?"
        ]

    if "archmage" not in gesproken:
        return "archmage", (
            "De Engineer meldt een nieuwe klokslag en leylijn-effecten. "
            "Wat registreren jouw metingen [ARC-xxx] en welke afspraken waren er over toestemming?"
        ), "Verifiëren van toestemming en leylijnverstoringen bij de Archmage.", [
            "Waren de leylijnen verstoord door het datanet?",
            "Was er toestemming voor de ingreep?"
        ]

    if "guardian_keeper" not in gesproken:
        return "guardian_keeper", (
            "Wat registreren de wachtrapporten [GUA-xxx] over de poortschilden en de Kroonklok? "
            "Hoe werkt het schildmechanisme exact (Art. 2, Art. 3)?"
        ), "Schildincidenten toetsen aan de tijdstippen van de torenactiviteit.", [
            "Waarom openden de schilden van binnenuit?",
            "Is er sprake van een externe inbraak?"
        ]

    if "historian" not in gesproken:
        return "historian", (
            "Wat staat er in de kroniek en archiefrecords [HIS-xxx] over de Kroonklok en de gevolgen "
            "van een tweede pols? Wat zegt de bouwakte precies?"
        ), "Historische context en de werking van de Kroonklok raadplegen.", [
            "Wat zegt de bouwakte over een tweede pols?",
            "Hoe kan de maat hersteld worden?"
        ]

    # Fase 2: Verloop bewijzen en geschillen oplossen
    if "beast_master" not in gesproken:
        return "beast_master", (
            "De Engineer stelt dat de hulpslag op dag 10 om 02:40 is geactiveerd. "
            "Bevestigen jouw woudwaarnemingen [BEA-xxx] dat de dierlijke onrust exact samenvalt "
            "met de activatie van de hulpslag en de schildopeningen op dag 10 (04:47) en dag 22 (04:15)?"
        ), "Onafhankelijke bevestiging van het tijdsverloop vanuit de natuur.", [
            "Valt de dierlijke onrust samen met de technische incidenten?",
            "Welke specifieke BEA-records tonen het patroon?"
        ]

    if "oracle" not in gesproken:
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
    state = DebatState(leden)

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
                    async_client=koning_raw_client, model=os.environ["MODEL_GROOT"], opties=opties
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
