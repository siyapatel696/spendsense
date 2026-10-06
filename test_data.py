from data_processing import (
    load_data,
    create_monthly_summary,
    create_baskets
)

from apriori_engine import (
    run_apriori,
    get_budget_rules
)


transactions, budgets = load_data()

monthly = create_monthly_summary(
    transactions,
    budgets
)

baskets = create_baskets(
    transactions,
    monthly
)

frequent_itemsets, rules = run_apriori(
    baskets,
    min_support=0.2,
    min_confidence=0.5
)

print("\nFREQUENT ITEMSETS")

print(frequent_itemsets)


print("\nALL ASSOCIATION RULES")

print(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]
)


budget_rules = get_budget_rules(rules)

print("\nBUDGET LEAKAGE RULES")

print(
    budget_rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]
)