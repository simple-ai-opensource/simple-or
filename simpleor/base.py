# base.py

from abc import ABCMeta, abstractmethod
import logging
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LOGGING_LEVELS = ["debug", "info", "critical"]
PROJECT_DIRECTORY = Path(os.path.dirname(os.path.realpath(__file__))).parent


def _configure_logger(verbose: str):
    """
    Define logs formats.

    Args:
        verbose: logging level
    """
    if verbose == "debug":
        level = logging.DEBUG
    elif verbose == "info":
        level = logging.INFO
    elif verbose == "critical":
        level = logging.CRITICAL
    else:
        raise ValueError(
            f"verbose {verbose} not recognized. choose from {LOGGING_LEVELS}."
        )

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)-8s | Process: %(process)d | %(name)s:%("
        "funcName)s:%(lineno)d - %(message)s",
        level=level,
    )

    logger.setLevel(level)


class Solver(metaclass=ABCMeta):
    @abstractmethod
    def validate_input(self):
        pass

    @abstractmethod
    def set_problem(self):
        pass

    @abstractmethod
    def _set_variables(self):
        pass

    @abstractmethod
    def _set_objective(self):
        pass

    @abstractmethod
    def _set_constraints(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_status(self):
        pass

    @abstractmethod
    def get_objective_value(self):
        pass

    @abstractmethod
    def get_solution(self):
        pass


class ProblemGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self):
        pass
