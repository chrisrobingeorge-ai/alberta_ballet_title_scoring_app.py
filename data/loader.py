from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_history_sales(csv_name: str = "history_city_sales.csv") -> pd.DataFrame:
    """Load historical show-level sales (Calgary/Edmonton)."""
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # Expected columns: show_title, single_tickets_calgary, single_tickets_edmonton,
    # subscription_tickets_calgary, subscription_tickets_edmonton, total_single_tickets
    return df
