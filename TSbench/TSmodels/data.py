"""Define the Data type used for models."""
from typing import Union
import pandas as pd
import numpy as np
from TSbench.TSdata import TSloader

Data = Union[list, np.ndarray, pd.DataFrame]

def size(data: Data):
    if type(data) is np.ndarray:
        return np.size(data, 0)
    if type(data) is list:
        # list of arrays or list of list
        if len(data) > 0 and (type(data[0]) is np.ndarray or type(data[0]) is list):
            return np.size(data[0], 0)
        else: # non-nested list
            return len(data)
    if type(data) is pd.DataFrame:
        return data.shape[0]
    else:
        raise ValueError("Data is of the wrong type to format")
