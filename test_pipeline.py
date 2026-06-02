import sys
sys.path.insert(0, ".")
from nlp.pipeline import NLPPipeline

db_path = "database/uploads/customers-100.db"
pipeline = NLPPipeline(db_path=db_path)

query = "Show first 10 rows from customers_100"
result = pipeline.process(query)

print("Query:", query)
print("Matched tables:", result["schema_links"]["matched_tables"])
print("SQL hints:", result["sql_hints"])
print("Schema tables:", list(pipeline.schema_linker.schema.keys()))
