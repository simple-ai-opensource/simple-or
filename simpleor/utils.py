import os
from pathlib import Path

PROJECT_DIRECTORY = Path(os.path.dirname(os.path.realpath(__file__))).parent


def _check_if_type(variable, kind, error_message: str):
    if not isinstance(variable, kind):
        raise ValueError(
            f"{error_message} type(variable) = {type(variable)}, should be {kind}"
        )
