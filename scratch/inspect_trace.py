import json

with open("trace.json", encoding="utf-8") as f:
    t = json.load(f)

beurten = t.get("beurten", [])
berichten = [b for b in beurten if b.get("soort") == "bericht" and b.get("spreker") != "koning"]
print(f"Raadslid beurten gebruikt: {len(berichten)}")
print(f"Totaal beurten in trace: {len(beurten)}")
print(f"Status in trace: {t.get('status')}")

routeringen = [b for b in beurten if b.get("soort") == "routering"]
for r in routeringen:
    ov = r.get("open_vragen", [])
    print(f"Routering beurt {r.get('nr')}: aan={r.get('aan')} (open_vragen: {len(ov)})")

print(f"\nGeschillen ({len(t.get('geschillen', []))}):")
for i, g in enumerate(t.get("geschillen", []), 1):
    print(f"\n--- Geschil {i}: {g.get('onderwerp')} ---")
    print(f"  Standpunten: {json.dumps(g.get('standpunten'), ensure_ascii=False)}")
    print(f"  Uitkomst: {g.get('uitkomst')}")
    print(f"  Bewijs: {g.get('bewijs')}")
