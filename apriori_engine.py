import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(
    baskets,
    min_support=0.2,
    min_confidence=0.5
):

    transactions = baskets["Items"].tolist()

    encoder = TransactionEncoder()

    encoded_data = encoder.fit(
        transactions
    ).transform(
        transactions
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.columns_
    )

    frequent_itemsets = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    return frequent_itemsets, rules


def get_budget_rules(rules):

    budget_rules = rules[
        rules["consequents"].apply(
            lambda x: "OVER_BUDGET" in x
        )
    ]

    return budget_rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )