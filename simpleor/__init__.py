"""Top-level package for simpleor."""
import os
from pathlib import Path

import simpleor.base
import simpleor.scheduler
import simpleor.cli

__author__ = """Lennart Damen"""
__email__ = "lennart.damen.ai@gmail.com"
__version__ = "0.0.4"

PROJECT_DIRECTORY = Path(os.path.dirname(os.path.realpath(__file__))).parent
