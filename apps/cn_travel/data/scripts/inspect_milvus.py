#!/usr/bin/env python3
"""
Inspect Milvus database contents
"""
from pymilvus import connections, Collection
import json
from data.paths import PATHS

# Connect to database
connections.connect(uri=str(PATHS.milvus_db))

# Get collection
collection = Collection("travel_guides")
collection.load()

# Get basic stats
print("=" * 60)
print("📊 Database Statistics")
print("=" * 60)
print(f"Collection Name: {collection.name}")
print(f"Total Entities: {collection.num_entities}")
print(f"Schema: {collection.schema}")
print()

# Query some samples
print("=" * 60)
print("📝 Sample Records (First 5)")
print("=" * 60)
results = collection.query(
    expr="",
    output_fields=["city_code", "city_name", "province_name", "content"],
    limit=5
)

for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['province_name']} - {result['city_name']} (Code: {result['city_code']})")
    print(f"   Content preview: {result['content'][:200]}...")
    print()

# Get unique provinces
print("=" * 60)
print("🗺️  Province Distribution")
print("=" * 60)
all_results = collection.query(
    expr="",
    output_fields=["province_name"],
    limit=1000
)

provinces = {}
for result in all_results:
    province = result['province_name']
    provinces[province] = provinces.get(province, 0) + 1

for province, count in sorted(provinces.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{province}: {count} 条")

print(f"\nTotal provinces: {len(provinces)}")
print(f"Total records queried: {len(all_results)}")

