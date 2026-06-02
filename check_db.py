import sqlite3
conn = sqlite3.connect("database/uploads/customers-100.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)
for (t,) in tables:
    cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
    print(f"\n  {t}:")
    for c in cols:
        print(f"    {c[1]} ({c[2]})")
    count = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    print(f"    → {count} rows")
conn.close()
