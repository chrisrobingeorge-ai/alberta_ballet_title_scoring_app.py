from pathlib import Path
import pandas as pd

CONFIG_DIR = Path(__file__).parent

def load_feature_inventory() -> pd.DataFrame:
    """Load the feature inventory CSV as a DataFrame."""
    return pd.read_csv(CONFIG_DIR / "ml_feature_inventory_alberta_ballet.csv")

def load_join_keys() -> pd.DataFrame:
    """Load the join keys CSV as a DataFrame."""
    return pd.read_csv(CONFIG_DIR / "ml_join_keys_alberta_ballet.csv")

def load_data_sources() -> pd.DataFrame:
    """Load the data sources CSV as a DataFrame."""
    return pd.read_csv(CONFIG_DIR / "ml_data_sources_alberta_ballet.csv")

def load_pipelines() -> pd.DataFrame:
    """Load the pipelines CSV as a DataFrame."""
    return pd.read_csv(CONFIG_DIR / "ml_pipelines_alberta_ballet.csv")

def load_leakage_audit() -> pd.DataFrame:
    """Load the leakage audit CSV as a DataFrame."""
    return pd.read_csv(CONFIG_DIR / "ml_leakage_audit_alberta_ballet.csv")

def load_modelling_tasks() -> pd.DataFrame:
    """Load the modelling tasks CSV as a DataFrame."""
    return pd.read_csv(CONFIG_DIR / "ml_modelling_tasks_alberta_ballet.csv")
