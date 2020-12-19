import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)-8s | Process: %(process)d | %(name)s:%("
    "funcName)s:%(lineno)d - %(message)s",
    level=logging.DEBUG,
)

LOGGING_LEVELS = ["debug", "info", "critical"]


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
