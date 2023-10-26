"""Define the Data type used for models."""
from typing import Union
import pandas as pd
import numpy as np
from TSbench.TSdata import TSloader

Data = Union[np.ndarray, pd.DataFrame, TSloader]
