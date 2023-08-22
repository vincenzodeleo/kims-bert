from yaml import safe_load
from yaml import YAMLError
from typing import Optional
from typing import Any


class ProblemTypeError(Exception):
    """Custom Exception for problem_type parameter."""
    pass


def read_yaml(filepath: str, key_name: Optional[Any] = None) -> Optional[dict]:
    """
    Read yaml configuration data.

    Parameters
    ----------
    filepath : str
        input config filepath
    key_name : str, default=None
        get specific configuration if there are multiple keys in the file yaml

    Returns
    -------
    dict or None
    """
    with open(file=filepath, mode='r') as file:
        try:
            if key_name:
                return read_yaml(filepath)[key_name]
            else:
                return safe_load(file)
        except YAMLError as error:
            print(error)
