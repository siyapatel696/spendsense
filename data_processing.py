import pandas as pd


def load_data():

    transactions = pd.read_csv("data/transactions.csv")
    budgets = pd.read_csv("data/budgets.csv")

    transactions["Date"] = pd.to_datetime(transactions["Date"])

    transactions["Month"] = (
        transactions["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    return transactions, budgets


def create_monthly_summary(transactions, budgets):

    monthly = (
        transactions
        .groupby(["User_ID", "Month"])["Amount"]
        .sum()
        .reset_index()
    )

    monthly = monthly.merge(
        budgets,
        on=["User_ID", "Month"],
        how="left"
    )

    monthly["Budget_Status"] = monthly.apply(
        lambda row:
        "OVER_BUDGET"
        if row["Amount"] > row["Budget"]
        else "WITHIN_BUDGET",
        axis=1
    )

    return monthly


def create_baskets(transactions, monthly):

    category_baskets = (
        transactions
        .groupby(["User_ID", "Month"])["Category"]
        .apply(set)
        .reset_index()
    )

    baskets = category_baskets.merge(
        monthly[["User_ID", "Month", "Budget_Status"]],
        on=["User_ID", "Month"],
        how="left"
    )

    baskets["Items"] = baskets.apply(
        lambda row: row["Category"] | {row["Budget_Status"]},
        axis=1
    )

    return baskets[["User_ID", "Month", "Items"]]