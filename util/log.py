"""
Shared logging setup: timestamp and level on every message.
Uses the standard library logging module (no extra deps).
"""
import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure the root logger so that log messages include date/time and level.
    Idempotent: safe to call multiple times; only configures once.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(format_string, datefmt=date_format)
    handler.setFormatter(formatter)
    root.addHandler(handler)
