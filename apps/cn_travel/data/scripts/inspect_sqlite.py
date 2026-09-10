#!/usr/bin/env python3
"""
Inspect Milvus SQLite database structure
"""
import sqlite3
from data.paths import PATHS

# Connect to database
conn = sqlite3.connect(str(PATHS.milvus_db))
cursor = conn.cursor()

# List all tables
print("=" * 60)
print("📋 Tables in Database")
print("=" * 60)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(f"  - {table[0]}")
print()

# For each table, show schema and row count
for table in tables:
    table_name = table[0]
    print("=" * 60)
    print(f"📊 Table: {table_name}")
    print("=" * 60)
    
    # Get schema
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  {col[1]}: {col[2]}")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    print(f"\nRow count: {count}")
    
    # Show sample data (first 2 rows)
    if count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 2;")
        rows = cursor.fetchall()
        print(f"\nSample data (first {min(2, count)} rows):")
        for i, row in enumerate(rows, 1):
            print(f"  Row {i}: {str(row)[:200]}{'...' if len(str(row)) > 200 else ''}")
    print()

conn.close()

