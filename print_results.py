import json

with open("phase2_results.json") as f:
    data = json.load(f)

for d in data["details"]:
    q = d["query"]
    intent = d["intent_hint"]
    tables = d["tables"]
    passed = d["passed"]
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Q{d['id']}: {q}")
    print(f"       Intent: {intent} | Tables: {tables}")
    if d.get("aggregations"):
        print(f"       Aggregations: {d['aggregations']}")
    if d.get("filters"):
        print(f"       Filters: {d['filters']}")
    if d.get("order"):
        print(f"       Order: {d['order']}")
    print(f"       Keywords: {d['keywords']}")
    print()

print(f"TOTAL: {data['passed']}/{data['total']} | ALL PASSED: {data['all_passed']}")
