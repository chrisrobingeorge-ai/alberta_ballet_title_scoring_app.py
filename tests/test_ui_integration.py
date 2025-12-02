"""
Smoke test: verifies UI can call forecast service predict() without error.
"""

import importlib

def test_ui_calls_forecast():
    # Import the forecast service
    svc = importlib.import_module("service.forecast")
    assert hasattr(svc, "predict"), "predict() not found in service.forecast"

    # Simulate UI call
    out = svc.predict("Coppelia", "YYC", "2025-10-17T19:30:00-06:00")
    assert isinstance(out, dict)
    assert "point" in out and "interval" in out and "drivers" in out
    assert {"p10","p50","p90"} <= set(out["interval"].keys())
