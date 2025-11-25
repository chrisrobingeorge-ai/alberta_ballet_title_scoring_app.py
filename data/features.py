import pandas as pd
from config.registry import load_feature_inventory


def apply_registry_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally map raw columns to registry names if needed."""
    # Example mapping (keep simple unless you need it)
    # return df.rename(columns={"total_single_tickets": "total_single_tickets"})
    return df


def derive_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple derived features based on registry guidance."""
    out = df.copy()
    # Example: total_subscription_tickets, total_tickets_all
    if {"subscription_tickets_calgary", "subscription_tickets_edmonton"}.issubset(out.columns):
        out["total_subscription_tickets"] = (
            out["subscription_tickets_calgary"].fillna(0) +
            out["subscription_tickets_edmonton"].fillna(0)
        )
    if {"total_single_tickets", "total_subscription_tickets"}.issubset(out.columns):
        out["total_tickets_all"] = (
            out["total_single_tickets"].fillna(0) + out["total_subscription_tickets"].fillna(0)
        )
    return out


def get_feature_list(theme_filters=None, status=None) -> list[str]:
    """Pull model feature names from the registry."""
    inv = load_feature_inventory()
    df = inv.copy()
    if theme_filters:
        df = df[df["Theme"].isin(theme_filters)]
    if status:
        df = df[df["Status"].str.contains(status, case=False)]
    return df["Feature Name"].dropna().unique().tolist()
