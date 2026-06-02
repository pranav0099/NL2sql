"""
Test NL2SQL pipeline with NOISY queries:
  - Wrong spellings
  - Bad grammar
  - Informal language
  
Verifies that the system still produces correct SQL
even when the user types messily.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.nl2sql_engine import NL2SQLEngine

# Test queries: (noisy input, expected SQL keyword/pattern)
NOISY_TESTS = [
    # === MISSPELLINGS ===
    ("shwo all staf",                       "SELECT", "staff"),
    ("lst all hotles",                      "SELECT", "hotels"),
    ("dispaly all guets",                   "SELECT", "guests"),
    ("shw all roms",                        "SELECT", "rooms"),
    ("show satff with salry above 50000",   "salary", "> 50000"),
    ("avrage salery of staf",               "AVG",    "salary"),
    ("cont all staf",                       "COUNT",  "staff"),
    ("totl payement amount",                "SUM",    "amount"),
    ("higest salry",                        "MAX",    "salary"),
    ("lowst salry",                         "MIN",    "salary"),
    
    # === BAD GRAMMAR ===
    ("show me the staff",                   "SELECT", "staff"),
    ("i want see all hotels",               "SELECT", "hotels"),
    ("plz show bookings",                   "SELECT", "bookings"),
    ("how much staff is there",             "COUNT",  "staff"),
    ("staff salary more then 50000",        "salary", "50000"),
    ("gimme all payments",                  "SELECT", "payments"),
    ("tell me average salary",              "AVG",    "salary"),
    ("whats the total payment",             "SUM",    ""),
    
    # === WRONG SENTENCE STRUCTURE ===
    ("all staff show",                      "SELECT", "staff"),
    ("salary staff above 50000",            "salary", "50000"),
    ("hotels in Mubmai show",               "hotel",  "Mumbai"),
    ("count how many bookings",             "COUNT",  "booking"),
    
    # === MIXED TYPOS + BAD GRAMMAR ===
    ("pls shw staf earning more then 60000","salary", "60000"),
    ("i want all hotles in Dehli",          "hotel",  "Delhi"),
    ("give me avrage salry",                "AVG",    "salary"),
    ("can u show staf in Kitchn",           "staff",  "Kitchen"),
]

def main():
    print("=" * 65)
    print("  NOISY QUERY TEST — Misspellings + Bad Grammar")
    print("=" * 65)
    
    engine = NL2SQLEngine()
    session = engine.create_session()
    
    passed = 0
    failed = 0
    results = []
    
    for noisy_q, expect_kw1, expect_kw2 in NOISY_TESTS:
        result = engine.query(noisy_q, session)
        sql = (result.get("sql") or "").upper()
        success = result.get("success", False)
        
        # Check if SQL contains expected keywords
        kw1_ok = expect_kw1.upper() in sql
        kw2_ok = expect_kw2.upper() in sql if expect_kw2 else True
        test_pass = success and kw1_ok and kw2_ok
        
        status = "PASS" if test_pass else "FAIL"
        if test_pass:
            passed += 1
        else:
            failed += 1
        
        results.append((status, noisy_q, result.get("sql", ""), success))
        
        print(f"\n  [{status}] {noisy_q}")
        print(f"       SQL: {result.get('sql', 'NONE')}")
        if not test_pass:
            print(f"       Expected keywords: {expect_kw1}, {expect_kw2}")
            print(f"       Success: {success}, Error: {result.get('error', '')}")
    
    # Summary
    total = passed + failed
    rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {passed}/{total} passed ({rate:.0f}%)")
    print(f"{'=' * 65}")
    
    if failed > 0:
        print(f"\n  FAILED QUERIES:")
        for status, q, sql, success in results:
            if status == "FAIL":
                print(f"    - {q}")
                print(f"      SQL: {sql}")
    
    return passed, total

if __name__ == "__main__":
    main()
