from typing import Dict, Any

def predict(title: str, city: str, performance_dt: str) -> Dict[str, Any]:
    """
    Returns a forecast dictionary:
    {
        'point': int,
        'interval': {'p10': int, 'p50': int, 'p90': int},
        'drivers': [{'feature': str, 'impact': float}]
    }
    """
    # TODO: Load trained model from /ml/model.pkl
    # For now, return dummy values for testing
    return {
        'point': 300,
        'interval': {'p10': 250, 'p50': 300, 'p90': 350},
        'drivers': [{'feature': 'marketing_spend_7d_lag', 'impact': 68}]
    }
