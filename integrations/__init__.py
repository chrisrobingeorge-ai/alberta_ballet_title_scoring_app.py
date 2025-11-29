"""
Integrations Package

This package contains API clients and data normalization utilities for
pulling show data from Ticketmaster and Archtics ticketing systems,
and event data from PredictHQ for demand intelligence.

Modules:
    ticketmaster: Ticketmaster Discovery/Partner API client
    archtics: Archtics Reporting API client  
    normalizer: Data normalization to target CSV schema
    csv_exporter: CSV export with target column order
    predicthq: PredictHQ Events API client for demand intelligence
"""

from .ticketmaster import TicketmasterClient
from .archtics import ArchticsClient
from .normalizer import ShowDataNormalizer
from .csv_exporter import export_show_csv
from .predicthq import (
    PredictHQClient,
    PredictHQEvent,
    PredictHQFeatures,
    PredictHQError,
    PredictHQAuthError,
    get_predicthq_features_dict,
)

__all__ = [
    "TicketmasterClient",
    "ArchticsClient", 
    "ShowDataNormalizer",
    "export_show_csv",
    "PredictHQClient",
    "PredictHQEvent",
    "PredictHQFeatures",
    "PredictHQError",
    "PredictHQAuthError",
    "get_predicthq_features_dict",
]
