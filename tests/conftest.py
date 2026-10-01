"""Shared pytest setup: put src/ on the path and gate network tests.

Network tests (marker `network`) download from the Hugging Face Hub. They are skipped
unless ASPECTD_NETWORK_TESTS=1; CI additionally deselects them with -m "not network".
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: downloads from the Hugging Face Hub "
                   "(skipped unless ASPECTD_NETWORK_TESTS=1)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ASPECTD_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set ASPECTD_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
