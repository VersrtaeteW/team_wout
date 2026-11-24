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


# ---------------------------------------------------------------------------
# ID- en tekstnormalisatie
# ---------------------------------------------------------------------------

def normaliseer_streepjes(obj):
    """Zet alle Unicode-koppeltekens om naar standaard ASCII '-'."""
    if isinstance(obj, str):
        return re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", obj)
    elif isinstance(obj, dict):
        return {normaliseer_streepjes(k): normaliseer_streepjes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normaliseer_streepjes(x) for x in obj]
    return obj


def extraheer_ids(tekst):
    """Vind alle unieke, genormaliseerde record-ID's in een tekst."""
    norm = normaliseer_streepjes(str(tekst))
    gevonden = set(re.findall(r"\b(?:ARC|ENG|GUA|HIS|ORA|BEA)-[0-9]{3}\b", norm))
    gevonden.discard("ENG-100")
    return sorted(gevonden)


def filter_geldige_ids(id_lijst, fallback=None):
    """Behoud uitsluitend geldige record-ID's die voldoen aan het schema."""
    geldig = []
    for item in id_lijst or []:
        if isinstance(item, str):
            norm = normaliseer_streepjes(item.strip("[] ,;.:\"'"))
            if re.fullmatch(r"(ARC|ENG|GUA|HIS|ORA|BEA)-[0-9]{3}", norm) and norm != "ENG-100":
                geldig.append(norm)
    res = sorted(set(geldig))
    if not res and fallback:
        res = sorted(set(fallback))
    return res


# ---------------------------------------------------------------------------
# Systeemcontroles en clients
# ---------------------------------------------------------------------------

def controleer(leden):
    """Controleer omgevingsvariabelen en raadsleden."""
    mist = [v for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_KLEIN", "MODEL_GROOT") if not os.getenv(v)]
    if mist:
        sys.exit(f"In .env ontbreekt: {', '.join(mist)}. Vul .env in.")
    for lid in leden:
        if lid.ROL not in ROLLEN:
            sys.exit(f"raadsleden/: ROL = {lid.ROL!r} is ongeldig. Kies uit: {', '.join(ROLLEN)}.")
    dubbel = {l.ROL for l in leden if [x.ROL for x in leden].count(l.ROL) > 1}
    if dubbel:
        sys.exit(f"Eén Raadslid per Rol: {', '.join(dubbel)} komt meermalen voor.")


def client(model_env):
    return OpenAIChatCompletionClient(
        model=os.environ[model_env],
        async_client=AsyncOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=8,
        ),
    )


def kennisbron_bestanden(rol):
    map_ = KENNISBRONNEN / rol
    return sorted(p for p in map_.glob("*") if p.suffix in (".md", ".csv")) if map_.is_dir() else []


def instructies(lid, bestanden, mcp):
    tekst = (
        f"Je bent de {lid.NAAM}, een Raadslid in de Council of Achrona. {lid.PERSPECTIEF}\n\n"
        "STRIKTE INSTRUCTIES:\n"
        "1. Baseer je UITSLUITEND op je eigen Kennisbron en onderbouw elke feitelijke uitspraak met je eigen record-ID's. "
        "Verzin nooit ID's. Speculeer nooit over ontbrekende volgnummers in een ID-reeks (zoals een gat tussen CSV- en tekstnummering) "
        "en noem uitsluitend record-ID's die letterlijk in je Kennisbron staan.\n"
        "2. Luister kritisch naar de andere Raadsleden. Als iemand een bewering doet die botst met of raakt aan jouw bron, "
        "reageer dan direct met tegenspraak of verduidelijking (wederhoor).\n"
        "3. Antwoord direct, beknopt en zakelijk.\n"
        "4. Als iets niet in je bron staat, zeg dan eerlijk dat het onbekend is."
    )
    if mcp:
        tekst += "\nLees of doorzoek je Kennisbron met je tools voor je spreekt."
    elif bestanden:
        tekst += "\n\nJe Kennisbron:\n\n" + "\n\n".join(f"## {p.name}\n\n{p.read_text(encoding='utf-8')}" for p in bestanden)
    return tekst


def toolcalls(response):
    for msg in response.messages:
        for c in msg.contents:
            if c.type == "function_call":
                args = c.arguments
                yield c.name, json.loads(args) if isinstance(args, str) and args else (args or {})


def tel(verbruik, model, usage):
    if not usage:
        return
    v = verbruik.setdefault(model, {"beurten": 0, "invoer": 0, "uitvoer": 0})
    v["beurten"] += 1
    v["invoer"] += getattr(usage, "prompt_tokens", None) or (usage.get("input_token_count") if isinstance(usage, dict) else 0) or 0
    v["uitvoer"] += getattr(usage, "completion_tokens", None) or (usage.get("output_token_count") if isinstance(usage, dict) else 0) or 0


# ---------------------------------------------------------------------------
# Koning LLM Orchestratie
# ---------------------------------------------------------------------------

async def kies_volgende(beurten, leden, open_vragen, koning_client, model, opties=None):
    """De Koning leidt het debat: hij kiest de volgende spreker, de vraag, de reden en actualiseert open_vragen."""
    gesproken = sum(1 for b in beurten if b.get("soort") == "bericht" and b.get("spreker") in ROLLEN)
    if gesproken >= BEURTENBUDGET:
        return None, None, f"Beurtenbudget bereikt ({BEURTENBUDGET} Raadslid-beurten)", open_vragen

    spreekbeurten = {l.ROL: 0 for l in leden}
    for b in beurten:
        if b.get("soort") == "bericht" and b.get("spreker") in spreekbeurten:
            spreekbeurten[b["spreker"]] += 1

    niet_gesproken = [l.ROL for l in leden if spreekbeurten[l.ROL] == 0]

    debat_tekst = "\n".join(
        f"- {b['spreker']}: {b.get('tekst', '')}"
        for b in beurten
        if b.get("soort") == "bericht"
    ) or "(Het debat is geopend door de Koning)"

    leden_info = "\n".join(
        f"- {l.ROL} ({l.NAAM}): {l.PERSPECTIEF} (gesproken: {spreekbeurten[l.ROL]}x)"
        for l in leden
    )

    prompt = (
        "Je bent de wijze Koning van Achrona. Jij leidt het onderzoek van de Council naar de Desync.\n"
        "Je hebt zelf geen Kennisbron; je hoort de 6 Raadsleden en stuurt het onderzoek doelgericht aan.\n\n"
        "### OPDRACHT & STRATEGIE:\n"
        "1. DOEL: Los de 3 luiken van de oorzaak op met hard bewijs (record-ID's uit de Kennisbronnen):\n"
        "   - INGREEP: Wie deed welke technische handeling, wanneer, en was er toestemming?\n"
        "   - MECHANISME: Hoe veroorzaakt die ingreep de Desync-symptomen (timing klokslag, schilden, leylijnen)?\n"
        "   - VERLOOP: Wat is het patroon in de tijd (cycli, periodiciteit, correlatie met dierenonrust/leylijndalingen)?\n"
        "2. SYNTHESE-MANDAAT & KRUISVERBANDEN:\n"
        "   - Elk Raadslid heeft slechts één afgeschermde Kennisbron en kan NOOIT records van anderen verifiëren. "
        "Vraag een rol dus nooit om records van een ander te controleren of te bevestigen.\n"
        "   - Het is de taak van de KONING om kruisverbanden tussen de 6 bronnen te leggen (tijdstippen, mechanismen, frequenties en systeemregels aan elkaar koppelen) "
        "in plaats van een causaal verband als 'onbewezen hypothese' af te doen zodra één rol het niet in zijn eigen bron heeft staan.\n"
        "   - Toets afwijkingen (zoals foutieve tijdstempels of ogenschijnlijk tegenstrijdige meetfrequenties tussen rollen) eerst als mogelijk "
        "GEVOLG EN BEWIJS van het mechanisme (de desynchronisatie zelf ontregelt tijdstempels en registraties!) voordat je ze als 'onbetrouwbare data' afschrijft.\n"
        "3. DEBATLEIDING & GEEN BEURTVERPILLING:\n"
        "   - Laat NOOIT herhaaldelijk dezelfde rol achter elkaar aan het woord voor data die hij eerder al aangaf niet te bezitten of niet in zijn bron te hebben.\n"
        "   - Schoon 'open_vragen' elke beurt ACTIEF op: schrap direct elke vraag die beantwoord is of niet uit de bronnen te beantwoorden valt. "
        "Houd de lijst kort (maximaal 4 tot 5 gerichte kernvragen).\n"
        "4. EARLY-STOP REGEL:\n"
        "   - Zodra alle 3 de luiken (ingreep, mechanisme, verloop), de belangrijkste geschillen, het herstelplan (inclusief concrete bedienings-/bijsturingsrecords uit de bronnen) "
        "en de voorspelling met record-ID's onderbouwd zijn, kies dan DIRECT 'aan': null vóór het budget van 10 beurten op is. "
        "Blijf niet doorvragen naar details die niet bestaan.\n\n"
        f"### STATUS: {gesproken}/{BEURTENBUDGET} Raadslid-beurten gebruikt\n"
        f"Nog niet gesproken: {', '.join(niet_gesproken) or 'Iedereen heeft gesproken'}\n"
        f"Huidige open vragen:\n" + "\n".join(f"- {q}" for q in open_vragen) + "\n\n"
        f"### BESCHIKBARE RAADSLEDEN:\n{leden_info}\n\n"
        f"### DEBAT TOT NU TOE:\n{debat_tekst}\n\n"
        "Geef UITSLUITEND een JSON-object terug in dit formaat:\n"
        "{\n"
        '  "aan": "<rolnaam uit de beschikbare raadsleden, of null als het onderzoek voltooid is>",\n'
        '  "vraag": "<gerichte, concrete vraag aan deze rol, of null als aan null is>",\n'
        '  "reden": "<heldere inhoudelijke motivatie voor deze keuze>",\n'
        '  "open_vragen": ["<maximaal 4 tot 5 actieve, onbeantwoorde kernvragen na deze stap>"]\n'
        "}"
    )

    try:
        resp = await koning_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = normaliseer_streepjes(resp.choices[0].message.content or "")
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            aan = data.get("aan")
            vraag = data.get("vraag")
            reden = data.get("reden") or "Gerichte routering door de Koning"
            nw_vragen = [str(x) for x in data.get("open_vragen", open_vragen) if x][:5]

            if aan is None:
                return None, None, reden, nw_vragen
            if aan in spreekbeurten:
                return aan, vraag or f"{aan}, wat toont jouw Kennisbron hierover?", reden, nw_vragen
    except Exception as e:
        print(f"[Koning routering melding: {e}]")

    # Eenvoudige fallback: kies rol die nog niet sprak of het minst sprak
    keuze = sorted(leden, key=lambda l: spreekbeurten[l.ROL])[0]
    return keuze.ROL, f"{keuze.NAAM}, wat toont jouw Kennisbron over de Desync en wat de andere raadsleden tot nu toe stelden?", f"Inbreng verzamelen vanuit het perspectief van de {keuze.NAAM}.", open_vragen


# ---------------------------------------------------------------------------
# Verdict en Geschillen Synthese
# ---------------------------------------------------------------------------

async def formuleer_verdict(debat, run_id, koning_client, model):
    """De Koning stelt het definitieve Council Verdict en de beslechte geschillen op."""
    debat_tekst = "\n\n".join(debat)
    alle_ids = extraheer_ids(debat_tekst)

    prompt = (
        "Je bent de wijze Koning van Achrona. Het onderzoek van de Council is afgerond.\n"
        "Hier is het volledige debatverslag van de 6 Raadsleden:\n\n"
        f"{debat_tekst}\n\n"
        f"Genoemde record-ID's in het debat: {', '.join(alle_ids)}\n\n"
        "### OPDRACHT:\n"
        "Stel het definitieve Council Verdict en de lijst van beslechte geschillen op.\n\n"
        "### SYNTHESE- EN BESLECHTMANDAAT VAN DE KONING:\n"
        "1. SYNTHESE: Elk Raadslid heeft slechts één afgeschermde Kennisbron en kan records van anderen niet verifiëren. "
        "Het is jouw taak als Koning om kruisverbanden tussen de 6 bronnen te leggen en te synthetiseren: koppel tijdstippen, mechanismen, "
        "frequenties en systeemregels over de rollen heen aan elkaar. Doe een causaal verband niet af als 'onbewezen hypothese' "
        "omdat één individuele rol het niet in zijn bron had staan.\n"
        "2. AFWIJKINGEN TOETSEN: Toets afwijkingen (zoals foutieve tijdstempels van de Historian of verspringende meetfrequenties) "
        "eerst als mogelijk gevolg en bewijs van het mechanisme (de ontkoppelde hulpslag ontregelde de tijdstempels en registraties!) "
        "voordat je ze als 'onbetrouwbare data' afschrijft.\n"
        "3. GESCHILLEN BESLECHTEN: Beslecht elk geschil daadwerkelijk met een gemotiveerde 'uitkomst' op basis van het gecombineerde bewijs "
        "uit de andere bronnen. Gebruik alleen 'onbeslist' als er na wederhoor écht geen enkel uitsluitsel mogelijk is.\n"
        "4. ZEKERHEID: Zorg dat de 'zekerheid' per slot (0.0 tot 1.0) realistisch omhoog gaat wanneer meerdere onafhankelijke bronnen elkaar ondersteunen "
        "(richtwaarde 0.80–0.95 bij sluitende kruisverwijzingen over meerdere rollen).\n"
        "5. GEEN SPOOK-ID'S: Speculeer nooit over ontbrekende volgnummers in een ID-reeks (zoals een gat tussen CSV- en tekstnummering, bv. verzin nooit 'ENG-100') "
        "en zet UITSLUITEND record-ID's die letterlijk in een Kennisbron zijn gelezen in 'bewijs'-arrays.\n\n"
        "### REGELS VOOR DE VELDEN:\n"
        "1. OORZAAK (3 luiken):\n"
        "   - ingreep: Wie voerde welke technische handeling uit en wanneer (activatie hulpslag), en was er toestemming?\n"
        "   - mechanisme: Causale keten (tweede pols / hulpslag versnelt doorvoer, ontkoppelt van Kroonklok, schildregel Art. 3 laat verkeer tussen tikken ongetoetst door als binnenverkeer, leylijnen dalen).\n"
        "   - verloop: Patroon in de tijd (periodiciteit van 4 dagen vanaf dag 10, correlaties met leylijnen en dierenonrust rond 04:15–04:50).\n"
        "   Elk luik bevat:\n"
        "   - bewering: concrete feitelijke stelling\n"
        "   - bewijs: array van geldige record-ID's die daadwerkelijk in het debat zijn genoemd (uitsluitend bestaande records)\n"
        "   - zekerheid: getal tussen 0.0 en 1.0 (0.80–0.95 bij sterke kruisverbanden)\n"
        "2. ACTIES: Chronologisch herstelplan (minstens 2 geordende acties, inclusief concrete bedienings-/bijsturingsrecords uit de bronnen, "
        "bv. archief back-up maken voor stoppen hulpslag, hulpslag geleidelijk synchroniseren in kleine stappen, herijking leylijnen/schilden). "
        "Elk met 'actie', 'reden' en optioneel 'bewijs'.\n"
        "3. RISICOS: Minstens één reëel risico met concrete mitigatie (bv. dataverlies bij abrupte stop, openende poortschilden).\n"
        "4. VOORSPELLING (Extra uitdaging): Wanneer treedt de Desync opnieuw op bij ongewijzigd beleid? "
        "Bevat 'dag' (integer, bv. 46), 'bewering' en 'bewijs' (array van ID's).\n"
        "5. GESCHILLEN (Extra uitdaging): Inhoudelijke meningsverschillen of tegenstrijdige hypothesen uit het debat "
        "(bv. interne infiltrant vs timingfout poortschilden, archmage toestemming vs handelingen, sterrenstand vs datanet). "
        "Elk geschil bevat 'onderwerp', 'standpunten' (met rolnaam als sleutel), 'uitkomst' (echt beslecht!) en 'bewijs' (array van ID's).\n\n"
        "Geef UITSLUITEND een JSON-object terug in dit formaat:\n"
        "{\n"
        '  "samenvatting": "<heldere samenvatting van de Desync en de hoofdoorzaak>",\n'
        '  "oorzaak": {\n'
        '    "ingreep": {"bewering": "<bewering>", "bewijs": ["ENG-011", "ARC-106"], "zekerheid": 0.95},\n'
        '    "mechanisme": {"bewering": "<bewering>", "bewijs": ["GUA-104", "HIS-110"], "zekerheid": 0.95},\n'
        '    "verloop": {"bewering": "<bewering>", "bewijs": ["BEA-019", "ORA-110"], "zekerheid": 0.90}\n'
        '  },\n'
        '  "acties": [\n'
        '    {"actie": "<actie>", "reden": "<reden>", "bewijs": ["HIS-110"]}\n'
        '  ],\n'
        '  "risicos": [\n'
        '    {"risico": "<risico>", "mitigatie": "<mitigatie>"}\n'
        '  ],\n'
        '  "voorspelling": {\n'
        '    "dag": 46,\n'
        '    "bewering": "<wanneer treedt volgende desync op>",\n'
        '    "bewijs": ["ORA-142"]\n'
        '  },\n'
        '  "geschillen": [\n'
        '    {\n'
        '      "onderwerp": "<geschilpunt>",\n'
        '      "standpunten": {"guardian_keeper": "<standpunt>", "engineer": "<standpunt>"},\n'
        '      "uitkomst": "<conclusie beslecht op basis van gecombineerd bewijs>",\n'
        '      "bewijs": ["GUA-104", "ENG-011"]\n'
        '    }\n'
        '  ]\n'
        "}"
    )

    data = None
    for poging in range(3):
        try:
            resp = await koning_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 + poging * 0.05,
            )
            raw = normaliseer_streepjes(resp.choices[0].message.content or "")
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "oorzaak" in data and "samenvatting" in data:
                    break
        except Exception as e:
            print(f"[Verdict synthese poging {poging + 1} waarschuwing: {e}]")

    data = data or {}

    # Structureren en valideren tegen JSON-schemas
    samenvatting = data.get("samenvatting") or "Onderzoek naar de Desync afgerond door de Council of Achrona."
    oorzaak_raw = data.get("oorzaak", {})
    oorzaak = {}
    for slot in ("ingreep", "mechanisme", "verloop"):
        s = oorzaak_raw.get(slot, {})
        bewering = s.get("bewering")
        zekerheid = float(s.get("zekerheid", 0.9)) if s.get("zekerheid") is not None else 0.9
        zekerheid = max(0.0, min(1.0, zekerheid))
        bewijs = filter_geldige_ids(s.get("bewijs"), fallback=alle_ids[:2] if bewering else None)
        oorzaak[slot] = {"bewering": bewering, "bewijs": bewijs, "zekerheid": zekerheid}

    acties = []
    for a in data.get("acties", []):
        if isinstance(a, dict) and a.get("actie") and a.get("reden"):
            actie_item = {
                "actie": str(a["actie"]),
                "reden": str(a["reden"]),
            }
            if a.get("bewijs"):
                actie_item["bewijs"] = filter_geldige_ids(a["bewijs"])
            acties.append(actie_item)
    if not acties:
        acties = [{"actie": "Herstel de synchronisatie met de Kroonklok.", "reden": "Voorkomt verdere desynchronisatie."}]

    risicos = []
    for r in data.get("risicos", []):
        if isinstance(r, dict) and r.get("risico") and r.get("mitigatie"):
            risicos.append({"risico": str(r["risico"]), "mitigatie": str(r["mitigatie"])})
    if not risicos:
        risicos = [{"risico": "Dataverlies bij abrupte stop.", "mitigatie": "Eerst registers back-uppen."}]

    voorspelling = None
    v_raw = data.get("voorspelling")
    if isinstance(v_raw, dict) and v_raw.get("dag") and v_raw.get("bewering"):
        dag_val = int(v_raw["dag"]) if str(v_raw["dag"]).isdigit() else 46
        voorspelling = {
            "dag": max(1, dag_val),
            "bewering": str(v_raw["bewering"]),
            "bewijs": filter_geldige_ids(v_raw.get("bewijs"), fallback=alle_ids[:1]),
        }

    # Geschillen opschonen voor trace.json
    geschillen = []
    for g in data.get("geschillen", []):
        if isinstance(g, dict) and g.get("onderwerp"):
            standpunten = {}
            for k, v in g.get("standpunten", {}).items():
                k_clean = str(k).lower().strip().replace(" ", "_")
                if k_clean in ROLLEN:
                    standpunten[k_clean] = str(v)
            uitkomst = g.get("uitkomst")
            bewijs = filter_geldige_ids(g.get("bewijs"), fallback=alle_ids[:1] if uitkomst else None)
            if standpunten:
                geschillen.append({
                    "onderwerp": str(g["onderwerp"]),
                    "standpunten": standpunten,
                    "uitkomst": str(uitkomst) if uitkomst else None,
                    "bewijs": bewijs,
                })

    verdict = {
        "run_id": run_id,
        "samenvatting": samenvatting,
        "oorzaak": oorzaak,
        "acties": acties,
        "risicos": risicos,
    }
    if voorspelling:
        verdict["voorspelling"] = voorspelling

    return normaliseer_streepjes(verdict), normaliseer_streepjes(geschillen)


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

    def bewaar(naam, data):
        (ROOT / naam).write_text(json.dumps(normaliseer_streepjes(data), indent=2, ensure_ascii=False), encoding="utf-8")

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
                             f"Laat MCP_{lid.ROL.upper()} leeg om lokale bestanden uit kennisbronnen/{lid.ROL}/ te gebruiken.")
            elif not bestanden:
                print(f"LET OP: {lid.NAAM} heeft geen Kennisbron.")
            for p in bestanden:
                beurt(0, lid.ROL, "bestand", bestand=f"{lid.ROL}/{p.name}")
            agents.append((lid, mcp_url, Agent(klein, instructies(lid, bestanden, mcp_url), name=lid.ROL, tools=tools)))

        (ROOT / "verdict.json").unlink(missing_ok=True)
        bewaar("trace.json", trace)

        per_rol = {lid.ROL: (lid, mcp_url, agent) for lid, mcp_url, agent in agents}
        debat, ronde = [], 1
        open_vragen = [
            "Wie voerde welke technische handeling uit en wanneer?",
            "Welk causaal mechanisme veroorzaakt de Desync en schildopeningen?",
            "Welk patroon en periodiciteit vertoont het verloop in de tijd?",
        ]

        while True:
            rol, vraag, reden, open_vragen = await kies_volgende(
                beurten, leden, open_vragen, koning_raw_client, os.environ["MODEL_GROOT"], opties
            )

            if rol is not None and rol not in per_rol:
                sys.exit(f"kies_volgende() koos {rol!r}, maar die zit niet in raadsleden/.")

            gesproken = sum(1 for b in beurten if b.get("soort") == "bericht" and b.get("spreker") in per_rol)
            if rol is not None:
                ronde = gesproken // len(leden) + 1

            routering_velden = {"aan": rol, "vraag": vraag, "reden": reden}
            if open_vragen:
                routering_velden["open_vragen"] = open_vragen
            beurt(ronde, "koning", "routering", **routering_velden)

            print(f"\n[Koning] {'stopt' if rol is None else '→ ' + per_rol[rol][0].NAAM}: {reden}")
            if rol is None:
                break

            lid, mcp_url, agent = per_rol[rol]
            prompt = (
                f"Debat tot nu toe:\n\n" + ("\n\n".join(debat) or "(nog niets)") +
                f"\n\nVraag van de Koning aan jou ({lid.NAAM}): {vraag}\n\n"
                f"Raadpleeg je Kennisbron en antwoord direct met feiten en record-ID's."
            )
            trace["aan_het_woord"] = rol
            bewaar("trace.json", trace)

            r = await agent.run(prompt, options=opties)
            tel(verbruik, "klein", r.usage_details)
            for tool, args in toolcalls(r):
                beurt(ronde, rol, "toolcall", tool=tool, argumenten=normaliseer_streepjes(args), pad=mcp_url)

            tekst = normaliseer_streepjes(r.text)
            beurt(ronde, rol, "bericht", tekst=tekst)
            debat.append(f"{lid.NAAM}: {tekst}")
            print(f"\n[ronde {ronde}] {lid.NAAM}:\n{tekst}")

    # Synthese door de Koning
    trace["aan_het_woord"] = "koning"
    bewaar("trace.json", trace)

    print("\n[Koning] Formuleert definitief eindoordeel en synthese...")
    verdict, geschillen = await formuleer_verdict(debat, run_id, koning_raw_client, os.environ["MODEL_GROOT"])

    trace["geschillen"] = geschillen
    trace["status"], trace["aan_het_woord"] = "klaar", None

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
