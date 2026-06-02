import json

data = json.load(open("data/processed/train.json", "r", encoding="utf-8"))
nl_set = set(d["question"].lower().strip() for d in data)

tests = [
    "show the cheapest product in each category",
    "which city has the most employees",
    "find customers who have never placed an order",
    "show departments where total salary exceeds 200000",
    "what percentage of orders are delivered",
    "show month over month sales growth",
    "find products that cost more than the average price",
    "which employee has the highest salary in each department",
    "show customers who joined in 2024 and spent more than 30000",
    "list cities where average order amount exceeds 5000",
]

print("--- Checking if test queries exist in training data ---")
for t in tests:
    status = "FOUND" if t.lower() in nl_set else "NOT FOUND"
    print(f"  [{status}]  {t}")
