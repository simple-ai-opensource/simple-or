import logging
import sys
import timeit
from typing import Callable, List
import functools
import pandas as pd

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)-8s | Process: %(process)d | %(name)s:%("
    "funcName)s:%(lineno)d - %(message)s",
    level=logging.DEBUG,
)

LOGGING_LEVELS = ["debug", "info", "warning", "error", "critical"]


def _configure_logger(verbose: str):
    """
    Define logs level and formats.

    Args:
        verbose: logging level
    """
    if verbose == "debug":
        level = logging.DEBUG
    elif verbose == "info":
        level = logging.INFO
    elif verbose == "warning":
        level = logging.INFO
    elif verbose == "error":
        level = logging.ERROR
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


class MonitorSingleton:

    __instance = None
    metrics: dict = {}

    def __init__(self):
        if MonitorSingleton.__instance is not None:
            raise TypeError(f"{self.__class__.__name__} is a singleton!")
        MonitorSingleton.__instance = self

    @staticmethod
    def get_instance():
        if MonitorSingleton.__instance is None:
            MonitorSingleton()
        return MonitorSingleton.__instance

    def _add_value_to_monitor_dict(self, key: str, value: object):
        if key not in self.metrics.keys():
            self.metrics[key] = [value]
        else:
            self.metrics[key] += [value]

    def _add_value_to_dataframe(
        self,
        dataframe_name: str,
        column_names: List[str],
        column_values: List,
        value: object,
    ):
        append_df = pd.DataFrame(
            columns=column_names + ["value"], data=[column_values + [value]]
        )
        if dataframe_name not in self.metrics.keys():
            self.metrics[dataframe_name] = append_df
        else:
            self.metrics[dataframe_name] = pd.concat(
                [self.metrics[dataframe_name], append_df]
            )

    def add_execution_time_to_monitor_dict(self, func: Callable, key):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = timeit.default_timer()
            result = func(*args, **kwargs)
            t1 = timeit.default_timer()
            self._add_value_to_monitor_dict(key=key, value=t1 - t0)
            return result

        return wrapper

    def add_execution_time_to_monitor_dataframe(
        self,
        func: Callable,
        dataframe_name: str,
        column_names: str,
        column_values: List,
    ):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = timeit.default_timer()
            result = func(*args, **kwargs)
            t1 = timeit.default_timer()
            self._add_value_to_dataframe(
                dataframe_name=dataframe_name,
                column_names=column_names,
                column_values=column_values,
                value=t1 - t0,
            )
            return result

        return wrapper
