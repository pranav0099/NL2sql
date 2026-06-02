import os
import sqlite3
import pandas as pd
import re
from pathlib import Path

# Paths
CSV_SRC_DIR = Path("c:/Users/prana/OneDrive/Desktop/projects/ns/database/uploads")
DOCS_CSV_DIR = Path("c:/Users/prana/OneDrive/Documents/csv")
UPLOADS_DIR = Path("c:/Users/prana/OneDrive/Desktop/projects/ns/database/uploads")
DB_PATH = Path("c:/Users/prana/OneDrive/Desktop/projects/ns/database/sample.db")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# List of files to copy from documents to uploads
files_to_copy = [
    "guests.csv",
    "hotels.csv",
    "payments.csv",
    "rooms.csv",
    "amenities.csv",
    "bookings.csv",
    "maintenance_logs.csv",
    "reviews.csv",
    "room_service_orders.csv",
    "staff.csv",
    "customers-100.csv"
]

print("Copying missing CSV files to uploads directory...")
for f_name in files_to_copy:
    src = DOCS_CSV_DIR / f_name
    dest = UPLOADS_DIR / f_name
    if src.exists():
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            pd.read_csv(src).to_csv(dest, index=False)
            print(f"  Copied {f_name} -> {dest}")
    else:
        print(f"  Warning: Source file {src} does not exist.")

# Connect to database
conn = sqlite3.connect(str(DB_PATH))
print(f"Connected to database: {DB_PATH}")

csv_files = [
    (UPLOADS_DIR / "staff.csv", "staff"),
    (UPLOADS_DIR / "hotels.csv", "hotels"),
    (UPLOADS_DIR / "guests.csv", "guests"),
    (UPLOADS_DIR / "rooms.csv", "rooms"),
    (UPLOADS_DIR / "bookings.csv", "bookings"),
    (UPLOADS_DIR / "payments.csv", "payments"),
    (UPLOADS_DIR / "reviews.csv", "reviews"),
    (UPLOADS_DIR / "maintenance_logs.csv", "maintenance_logs"),
    (UPLOADS_DIR / "amenities.csv", "amenities"),
    (UPLOADS_DIR / "room_service_orders.csv", "room_service_orders"),
    (UPLOADS_DIR / "hotel_bookings_updated_2024.csv", "hotel_bookings_updated_2024"),
    (UPLOADS_DIR / "crop_price_data.csv", "crop_price_data"),
    (UPLOADS_DIR / "customers-100.csv", "customers_100"),
]

for csv_path, table_name in csv_files:
    if not csv_path.exists():
        print(f"Skipping {csv_path.name} (not found)")
        continue
    
    # Read CSV robustly
    df = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception:
            continue
            
    if df is None:
        print(f"Error: Could not read {csv_path}")
        continue
        
    # Clean columns
    clean_cols = []
    col_counts = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        clean = re.sub(r"[^a-z0-9_]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        if not clean:
            clean = "column"
        if clean in col_counts:
            col_counts[clean] += 1
            clean = f"{clean}_{col_counts[clean]}"
        else:
            col_counts[clean] = 1
        clean_cols.append(clean)
    
    df.columns = clean_cols
    
    # Save to SQLite table
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Successfully loaded '{table_name}' table ({len(df)} rows, columns: {list(df.columns)})")

conn.close()
print("All tables successfully imported into sample.db!")
