"""Fraud detection on PaySim mobile transactions."""

from importlib.metadata import version

from fraud_detection.config import settings

# Read from the installed package metadata so pyproject.toml stays the single source.
__version__ = version("fraud-detection-mlops")

__all__ = ["__version__", "settings"]
