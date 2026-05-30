"""Define the Data type used for models."""

from typing import TypeVar, Union

import numpy as np
import pandas as pd

Data = TypeVar("Data", bound=np.ndarray | pd.DataFrame)
AnyData = Union[list, np.ndarray, pd.DataFrame]


def size(data: AnyData):
    """Return the length of the first axis of a time series data object.

    Args:
        data (AnyData): Input data as np.ndarray, list, or pd.DataFrame.

    Returns:
        int: Length along the first axis.

    Raises:
        ValueError: If ``data`` is not a supported type.

    """
    if isinstance(data, np.ndarray):
        return np.size(data, 0)
    if isinstance(data, list):
        # list of arrays or list of list
        if len(data) > 0 and (
            isinstance(data[0], np.ndarray) or isinstance(data[0], list)
        ):
            return np.size(data[0], 0)
        else:  # non-nested list
            return len(data)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.shape[0]
    else:
        raise ValueError("Data is of the wrong type to format")
