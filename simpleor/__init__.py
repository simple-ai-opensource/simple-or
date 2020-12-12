"""Top-level package for simpleor."""
import sys
import logging
import simpleor.base
import simpleor.scheduler
import simpleor.cli

__author__ = """Lennart Damen"""
__email__ = "lennart.damen@hotmail.com"
__version__ = "0.0.4"


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)-8s | Process: %(process)d | %(name)s:%("
    "funcName)s:%(lineno)d - %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(f"{__name__}")
