"""Pytest configuration and shared fixtures."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled."""
    run_integration = os.getenv("RUN_INTEGRATION_TESTS") or os.getenv("RUN_INTEGRATION")
    if run_integration and run_integration.lower() not in {"0", "false", "no"}:
        return

    skip_marker = pytest.mark.skip(
        reason="Integration test skipped (set RUN_INTEGRATION_TESTS=1 to run)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
