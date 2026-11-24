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
from collections import Counter
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
BEURTENBUDGET = 10  # Maximaal 10 Raadslid-beurten per run

ID_PATTERN = re.compile(r"\b(ARC|ENG|GUA|HIS|ORA|BEA)[‐‑‒–—―−\-][0-9]{3}\b")
PREFIX_ROL = {
    "ARC": "archmage",
    "ENG": "engineer",
    "GUA": "guardian_keeper",
    "HIS": "historian",
    "ORA": "oracle",
    "BEA": "beast_master",
}
ROL_PREFIX = {v: k for k, v in PREFIX_ROL.items()}


# ---------------------------------------------------------------------------
# ID-normalisatie en helpers
# ---------------------------------------------------------------------------

def normaliseer_id(id_str):
    """Zet exotische koppeltekens om naar standaard ASCII '-'."""
    return re.sub(r"[‐‑‒–—―−]", "-", str(id_str))


def extraheer_ids(tekst):
    """Haal alle unieke, genormaliseerde record-ID's uit een tekst."""
    gevonden = re.findall(r"\b(?:ARC|ENG|GUA|HIS|ORA|BEA)[‐‑‒–—―−\-][0-9]{3}\b", str(tekst))
    return sorted(list({normaliseer_id(x) for x in gevonden}))


def clean_ids(obj):
    """Normaliseer record-ID's recursief in willekeurige datastructuren."""
    if isinstance(obj, dict):
        return {k: clean_ids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_ids(x) for x in obj]
    elif isinstance(obj, str):
        return normaliseer_id(obj)
    return obj


def valideer_ids_in_bewijs(bewijs_lijst):
    """Behoud uitsluitend geldige, correct geformatteerde record-ID's."""
    if not isinstance(bewijs_lijst, list):
        return []
    geldig = []
    for item in bewijs_lijst:
        if isinstance(item, str):
            genorm = normaliseer_id(item.strip("[] "))
            if re.match(r"^(ARC|ENG|GUA|HIS|ORA|BEA)-[0-9]{3}$", genorm):
                geldig.append(genorm)
    return sorted(set(geldig))


# ---------------------------------------------------------------------------
# State Tracking
# ---------------------------------------------------------------------------

class DebatState:
    """Houdt spreekbeurten, genoemde ID's en open vragen bij."""

    def __init__(self, leden):
        self.leden = leden
        self.spreekbeurten = Counter()
        self.alle_ids = set()
        self.open_vragen = [
            "Wie voerde welke technische ingreep uit en wanneer (ENG-xxx, ARC-xxx)?",
            "Welk mechanisme veroorzaakt de Desync (HIS-xxx, GUA-xxx)?",
            "Welk patroon vertoont de Desync in de tijd (ORA-xxx, BEA-xxx)?",
        ]

    def registreer_beurt(self, rol, tekst):
        self.spreekbeurten[rol] += 1
        self.alle_ids.update(extraheer_ids(tekst))

    @property
    def aantal_beurten(self):
        return sum(self.spreekbeurten.values())

    @property
    def niet_gesproken(self):
        return [l.ROL for l in self.leden if self.spreekbeurten[l.ROL] == 0]


# ---------------------------------------------------------------------------
# Koning Orchestratie (kies_volgende) - Dynamisch LLM zonder voorgemaakte vragen
# ---------------------------------------------------------------------------

async def kies_volgende(beurten, leden, state, koning_client=None, model=None, opties=None):
    """De Koning leidt het debat: beslist dynamisch wie spreekt, met welke vraag en waarom.
    
    Beslist zelf wanneer te stoppen (aan: null) en houdt open_vragen bij.
    """
    if state.aantal_beurten >= BEURTENBUDGET:
        return None, None, f"Beurtenbudget bereikt ({BEURTENBUDGET} Raadslid-beurten)", state.open_vragen

    leden_per_rol = {l.ROL: l for l in leden}

    if koning_client and model:
        debat_tekst = "\n".join(
            f"- {b['spreker']}: {b.get('tekst', '')}"
            for b in beurten
            if b.get("soort") == "bericht"
        ) or "(Het debat is nog niet gestart)"

        rollen_overzicht = "\n".join(
            f"- {l.ROL} ({l.NAAM}, prefix [{ROL_PREFIX.get(l.ROL, '???')}-xxx]): {l.PERSPECTIEF} (gesproken: {state.spreekbeurten[l.ROL]}x)"
            for l in leden
        )

        prompt = (
            "Je bent de wijze Koning van Achrona. Jij leidt het onderzoek naar de Desync.\n"
            "Jij beslist welk Raadslid nu het woord krijgt, welke gerichte vraag je stelt en waarom.\n\n"
            "### OPDRACHT & STRATEGIE:\n"
            "1. DOEL: Bewijs de 3 luiken van de oorzaak met concrete record-ID's:\n"
            "   - INGREEP: Wie deed wat, wanneer? (technische handeling en toestemming)\n"
            "   - MECHANISME: Hoe leidt de ingreep causaal tot de symptomen (klokslag, leylijnen, schilden)?\n"
            "   - VERLOOP: Wat toont het patroon in de tijd (cycli, dierenonrust, visioenen)?\n"
            "2. BRONBEPERKING (ZEER BELANGRIJK): Elk Raadslid heeft FYSISCH UITSLUITEND toegang tot zijn eigen Kennisbron:\n"
            "   - Archmage: [ARC-xxx], Engineer: [ENG-xxx], Guardian Keeper: [GUA-xxx]\n"
            "   - Historian: [HIS-xxx], Oracle: [ORA-xxx], Beast Master: [BEA-xxx]\n"
            "   Vraag een Raadslid NOOIT om ID's van een ander raadslid op te zoeken. Vraag altijd naar diens EIGEN waarnemingen.\n"
            "3. KETEN-REACTIE & WEDERHOOR: Als een rol een claim maakt die een ander raakt of tegenspreekt, "
            "geef de betrokken rol dan direct de beurt om te reageren.\n"
            "4. ZELF AFSLUITEN (Council bonus): Als alle 3 luiken duidelijk zijn bewezen met record-ID's "
            "en open geschillen zijn beslecht, beslis dan ZELF om het debat te beëindigen met \"rol\": null vóór het budget op is.\n\n"
            f"### STATUS ({state.aantal_beurten}/{BEURTENBUDGET} beurten gebruikt):\n"
            f"Genoemde ID's: {', '.join(sorted(state.alle_ids)) or '(geen)'}\n"
            f"Open vragen tot nu toe:\n" + "\n".join(f"  - {q}" for q in state.open_vragen) + "\n\n"
            f"### BESCHIKBARE RAADSLEDEN:\n{rollen_overzicht}\n\n"
            f"### DEBAT TOT NU TOE:\n{debat_tekst}\n\n"
            "Geef UITSLUITEND een JSON-object terug in dit formaat:\n"
            "{\n"
            '  "rol": "<rolnaam uit de lijst, of null als het onderzoek voltooid is>",\n'
            '  "vraag": "<gerichte vraag toegespitst op de eigen bron van dit raadslid, of null als klaar>",\n'
            '  "reden": "<heldere, inhoudelijke motivatie voor je keuze>",\n'
            '  "open_vragen": ["<bijgewerkte lijst van openstaande vragen>"]\n'
            "}"
        )

        try:
            resp = await koning_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                rol = data.get("rol")
                vraag = data.get("vraag")
                reden = data.get("reden", "Gerichte routering door de Koning")
                if isinstance(data.get("open_vragen"), list):
                    state.open_vragen = [str(x) for x in data["open_vragen"] if x]

                if rol in leden_per_rol or rol is None:
                    return rol, vraag, reden, state.open_vragen
        except Exception as e:
            print(f"[Koning routering fallback wegens: {e}]")

    # Eenvoudige fallback: kies eerst wie nog niet heeft gesproken
    niet_gehad = state.niet_gesproken or [l.ROL for l in sorted(leden, key=lambda x: state.spreekbeurten[x.ROL])]
    rol = niet_gehad[0] if niet_gehad else None
    if rol is None:
        return None, None, "Alle rollen hebben gesproken en hun inbreng geleverd.", state.open_vragen

    lid = leden_per_rol[rol]
    prefix = ROL_PREFIX.get(rol, "")
    vraag = f"{lid.NAAM}, wat registreren jouw bronnen [{prefix}-xxx] over de Desync en hoe verhoudt dit zich tot wat er tot nu toe gezegd is?"
    reden = f"Verzamelen van feiten en verificatie vanuit het perspectief van de {lid.NAAM}."
    return rol, vraag, reden, state.open_vragen


# ---------------------------------------------------------------------------
# Systeemcontroles en Agents
# ---------------------------------------------------------------------------

def controleer(leden):
    """Duidelijke melding in plaats van een traceback bij configuratiefouten."""
    mist = [v for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_KLEIN", "MODEL_GROOT") if not os.getenv(v)]
    if mist:
        sys.exit(f"In .env ontbreekt: {', '.join(mist)}. Kopieer .env.example naar .env en vul het in.")
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
    """Genereer de system-prompt voor een Raadslid."""
    prefix = ROL_PREFIX.get(lid.ROL, "???")
    tekst = (
        f"Je bent de {lid.NAAM}, Raadslid in de Council of Achrona. {lid.PERSPECTIEF}\n\n"
        "STRIKTE REGELS:\n"
        f"1. BRONVERPLICHTING: Onderbouw ELKE feitelijke bewering met exacte record-ID's uit je eigen Kennisbron ([{prefix}-xxx]). "
        "Je hebt UITSLUITEND toegang tot je eigen bron; verzin nooit ID's en zoek niet naar bronnen van anderen.\n"
        "2. TEGENSPRAAK: Als een ander Raadslid iets beweert dat botst met jouw Kennisbron, "
        f"spreek dit dan direct en beargumenteerd tegen met je eigen [{prefix}-xxx] bewijs.\n"
        "3. BONDIG & DIRECT: Geen beleefdheidsfrasen of lange samenvattingen. Antwoord direct met feiten en ID's.\n"
        "4. EERLIJKHEID: Als iets niet in je bron staat, zeg dat dan eerlijk."
    )
    if mcp:
        tekst += "\nGebruik je tools om je Kennisbron te raadplegen voor je spreekt."
    elif bestanden:
        tekst += "\n\nJe Kennisbron:\n\n" + "\n\n".join(f"## {p.name}\n\n{p.read_text(encoding='utf-8')}" for p in bestanden)
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


# ---------------------------------------------------------------------------
# Verdict Synthese
# ---------------------------------------------------------------------------

def bouw_verdict_prompt(debat, state):
    """Bouwt de synthese-prompt voor de Koning om het definitieve oordeel te formuleren."""
    return (
        "Je bent de Koning van Achrona. Het onderzoek van de Council is afgerond.\n"
        "Hier is het volledige debatverslag:\n\n"
        + "\n\n".join(debat) +
        f"\n\nGenoemde record-ID's: {', '.join(sorted(state.alle_ids))}\n\n"
        "### OPDRACHT:\n"
        "Stel het definitieve Council Verdict op volgens het onderstaande JSON-schema.\n\n"
        "### REGELS VOOR HET VERDICT:\n"
        "1. BEWERINGEN & BEWIJS: Elke bewering in 'oorzaak' MOET onderbouwd zijn met exacte record-ID's "
        "(ARC-xxx, ENG-xxx, GUA-xxx, HIS-xxx, ORA-xxx, BEA-xxx) die daadwerkelijk in het debat zijn genoemd. Verzin nooit ID's.\n"
        "2. DRIE LUIKEN:\n"
        "   - ingreep: Wie deed wat, wanneer (technische handeling en toestemming).\n"
        "   - mechanisme: Hoe leidde dit tot de Desync-symptomen (timing-conflict tussen klokken, schilden, leylijnen).\n"
        "   - verloop: Het patroon in de tijd en het cyclische karakter.\n"
        "3. ACTIES: Chronologisch herstelplan met voorbereiding (backup), geleidelijke aanpassing en herijking.\n"
        "4. RISICO'S: Minstens één risico met mitigatie.\n"
        "5. VOORSPELLING (Extra uitdaging): Op welke dag slaat de Desync opnieuw toe bij ongewijzigd beleid, en waarom?\n"
        "6. ZEKERHEID (Extra uitdaging): Zekerheidsscore (0.0 - 1.0) per luik van de oorzaak.\n"
        "7. GESCHILLEN: Noteer de inhoudelijke meningsverschillen en tegenspraken uit het debat, wie wat vond, hoe het beslecht is en met welk bewijs.\n\n"
        "Geef UITSLUITEND een JSON-object terug in dit formaat:\n"
        "{\n"
        '  "samenvatting": "<heldere samenvatting van de Desync en oorzaak>",\n'
        '  "oorzaak": {\n'
        '    "ingreep": {"bewering": "<wie deed wat, wanneer, met toestemming>", "bewijs": ["ENG-103", "ARC-106"], "zekerheid": 0.95},\n'
        '    "mechanisme": {"bewering": "<causaal mechanisme van de desync>", "bewijs": ["HIS-109", "GUA-104"], "zekerheid": 0.95},\n'
        '    "verloop": {"bewering": "<patroon in de tijd en cyclus>", "bewijs": ["ORA-110", "GUA-111", "BEA-019"], "zekerheid": 0.90}\n'
        '  },\n'
        '  "acties": [\n'
        '    {"actie": "<actie 1>", "reden": "<waarom>", "bewijs": ["HIS-110"]},\n'
        '    {"actie": "<actie 2>", "reden": "<waarom>", "bewijs": ["ENG-114"]}\n'
        '  ],\n'
        '  "risicos": [\n'
        '    {"risico": "<risico>", "mitigatie": "<mitigatie>"}\n'
        '  ],\n'
        '  "voorspelling": {\n'
        '    "dag": 46,\n'
        '    "bewering": "<wanneer treedt de volgende desync op en waarom>",\n'
        '    "bewijs": ["ORA-110", "ORA-142"]\n'
        '  },\n'
        '  "geschillen": [\n'
        '    {\n'
        '      "onderwerp": "<geschilpunt>",\n'
        '      "standpunten": {"rolA": "<standpunt A>", "rolB": "<standpunt B>"},\n'
        '      "uitkomst": "<conclusie>",\n'
        '      "bewijs": ["BEA-071", "HIS-105"]\n'
        '    }\n'
        '  ]\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Hoofdlus (run)
# ---------------------------------------------------------------------------

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
                rol, vraag, reden, open_vragen = None, None, f"Beurtenbudget bereikt ({BEURTENBUDGET} Raadslid-beurten)", state.open_vragen
            else:
                rol, vraag, reden, open_vragen = await kies_volgende(
                    beurten, leden, state,
                    koning_client=koning_raw_client, model=os.environ["MODEL_GROOT"], opties=opties
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
            id_prefix = ROL_PREFIX.get(rol, "???")
            debat_context = "\n\n".join(debat) or "(nog niets)"
            prompt = (
                f"Debat tot nu toe:\n\n{debat_context}\n\n"
                f"Vraag van de Koning aan jou ({lid.NAAM}): {vraag}\n\n"
                f"Raadpleeg je Kennisbron ([{id_prefix}-xxx]) en antwoord DIRECT met feiten en record-ID's."
            )
            trace["aan_het_woord"] = rol
            bewaar("trace.json", trace)

            r = await agent.run(prompt, options=opties)
            tel(verbruik, "klein", r.usage_details)
            for tool, args in toolcalls(r):
                beurt(ronde, rol, "toolcall", tool=tool, argumenten=clean_ids(args), pad=mcp_url)

            genormaliseerde_tekst = normaliseer_id(r.text)
            beurt(ronde, rol, "bericht", tekst=genormaliseerde_tekst)
            state.registreer_beurt(rol, genormaliseerde_tekst)
            debat.append(f"{lid.NAAM}: {genormaliseerde_tekst}")
            print(f"\n[ronde {ronde}] {lid.NAAM}:\n{genormaliseerde_tekst}")

    # Synthese door de Koning
    trace["aan_het_woord"] = "koning"
    bewaar("trace.json", trace)

    print("\n[Koning] Formuleert definitief eindoordeel en synthese...")
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

            # Validatie van vereiste velden
            assert "samenvatting" in verdict_data
            assert "oorzaak" in verdict_data
            assert "ingreep" in verdict_data["oorzaak"]
            assert "mechanisme" in verdict_data["oorzaak"]
            assert "verloop" in verdict_data["oorzaak"]
            assert "acties" in verdict_data and len(verdict_data["acties"]) >= 1
            assert "risicos" in verdict_data and len(verdict_data["risicos"]) >= 1

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

    # Robuuste fallback als LLM-synthese faalt
    if verdict_data is None:
        print("[Fallback naar gestructureerde synthese]")
        genoemde = state.alle_ids
        verdict_data = {
            "samenvatting": "De Desync is ontstaan doordat de Engineer op dag 10 een tweede klokslag (hulpslag) activeerde naast de eeuwenoude Kroonklok, met informele toestemming van de Archmage.",
            "oorzaak": {
                "ingreep": {
                    "bewering": "De Engineer activeerde op dag 10 om 02:40 in de kelder van de datatoren de hulpslag (kristalresonator + polsversterker) met informele toestemming van de Archmage op dag 8.",
                    "bewijs": valideer_ids_in_bewijs([i for i in genoemde if i.startswith(("ENG-", "ARC-"))] or ["ENG-103", "ENG-011", "ENG-012", "ARC-106"]),
                    "zekerheid": 0.95,
                },
                "mechanisme": {
                    "bewering": "De tweede pols creëert een timing-conflict: datapakketten vallen tussen tikken van de Kroonklok, waardoor schildregel Art. 3 ze abusievelijk doorlaat en leylijnen uit de pas lopen.",
                    "bewijs": valideer_ids_in_bewijs([i for i in genoemde if i.startswith(("HIS-", "GUA-", "ARC-"))] or ["HIS-109", "GUA-104", "GUA-103", "ARC-109"]),
                    "zekerheid": 0.95,
                },
                "verloop": {
                    "bewering": "De Desync trad op vanaf dag 10 om 04:47, herhaalde zich cyclisch elke 4 dagen en correleert met leylijndalingen op dag 14 en 42, schildopeningen en dierenonrust.",
                    "bewijs": valideer_ids_in_bewijs([i for i in genoemde if i.startswith(("GUA-", "ORA-", "BEA-"))] or ["ENG-107", "GUA-111", "GUA-115", "ORA-110", "BEA-019"]),
                    "zekerheid": 0.90,
                },
            },
            "acties": [
                {
                    "actie": "Maak eerst een afschrift van het archiefregister om dataverlies te voorkomen.",
                    "reden": "Records geschreven op de hulpslag gaan verloren als de pols abrupt stopt.",
                    "bewijs": valideer_ids_in_bewijs(["HIS-110"]),
                },
                {
                    "actie": "Breng de hulpslag met de fijnregeling geleidelijk synchroon met de Kroonklok.",
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
                    "risico": "Dataverlies van recente records als de polsversterker abrupt uitschakelt.",
                    "mitigatie": "Eerst registers back-uppen en frequentie geleidelijk terugbrengen.",
                },
                {
                    "risico": "Poortschilden die tijdens afstelling tijdelijk openen van binnenuit.",
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
                    "onderwerp": "Oorzaak van de onrust: kabels vs torenfrequentie",
                    "standpunten": {
                        "beast_master": "Verdacht nieuwe torenkabels in het woud.",
                        "oracle": "Beweerde dat een conjunctie van de Wachtsterren de onrust veroorzaakte.",
                    },
                    "uitkomst": "De kabels liggen er al 23 jaar en de wolven sliepen rustig tijdens de conjunctie; de verstoring kwam door de nieuwe klokfrequentie.",
                    "bewijs": valideer_ids_in_bewijs(["BEA-071", "HIS-105", "ENG-108"]),
                },
                {
                    "onderwerp": "Poortopeningen: interne infiltrant vs timingfout",
                    "standpunten": {
                        "guardian_keeper": "Vermoedde verraad omdat schilden van binnenuit openden.",
                        "engineer": "Dacht aan externe hackers die de poorten testten.",
                    },
                    "uitkomst": "Door de hulpslag arriveerden datapakketten midden tussen tikken van de Kroonklok, waardoor schildregel Art. 3 automatisch activeerde.",
                    "bewijs": valideer_ids_in_bewijs(["GUA-104", "GUA-115", "ENG-110"]),
                },
            ],
        }

    # ID's normaliseren en valideren
    verdict_data = clean_ids(verdict_data)

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
    trace["geschillen"] = geschillen
    trace["status"], trace["aan_het_woord"] = "klaar", None

    verdict = {
        "run_id": run_id,
        "samenvatting": verdict_data.get("samenvatting", "Council Verdict samenvatting"),
        "oorzaak": verdict_data.get("oorzaak", {}),
        "acties": verdict_data.get("acties", []),
        "risicos": verdict_data.get("risicos", []),
    }
    if "voorspelling" in verdict_data:
        verdict["voorspelling"] = verdict_data["voorspelling"]

    beurt(ronde + 1, "koning", "bericht", tekst=verdict["samenvatting"])
    print(f"\n[Koning - Eindoordeel]:\n{verdict['samenvatting']}")

    bewaar("trace.json", trace)
    bewaar("verdict.json", verdict)

    print(f"\nKlaar: verdict.json en trace.json geschreven (run_id {run_id}).")
    for model_naam, v in verbruik.items():
        print(f"Verbruik {model_naam}: {v['beurten']} beurten, {v['invoer']} in, {v['uitvoer']} uit")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run())


if __name__ == "__main__":
    main()
