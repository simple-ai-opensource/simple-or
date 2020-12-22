"""Top-level package for simpleor."""

import simpleor.base
import simpleor.scheduler
import simpleor.cli
from simpleor.monitoring import _configure_logger

__author__ = """Lennart Damen"""
__email__ = "lennart.damen.ai@gmail.com"
__version__ = "0.0.7"

_configure_logger(verbose="info")
