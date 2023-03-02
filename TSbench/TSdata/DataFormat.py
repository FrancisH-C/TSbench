"""Example of data format to a TimeSeries DataFrame."""
import pandas as pd
import numpy as np


def df_to_TSdf(df, ID=None, timestamp=None, dim_label=None):
    """Convert a pandas DataFrame into a TimeSeries DataFrame.

    Keeps data already in DataFrame

    """
    # dim
    df = df.copy()
    if "dim" in df.columns:
        dim_label = set(df["dim"])
    else:
        if dim_label is None:
            dim_label = ["0"]
        T = df.shape[0] // len(dim_label)
        dim = np.array([dim_label for _ in range(T)]).flatten()
        df["dim"] = dim

    # timestamp
    if "timestamp" not in df.columns:
        if timestamp is None:
            T = df.shape[0] // len(dim_label)
            timestamp = list(map(str, range(0, T)))

        timestamp = np.transpose(
            np.array([timestamp for _ in range(len(dim_label))])
        ).flatten()
        df["timestamp"] = timestamp

    # ID
    if ID is not None:
        df["ID"] = ID
    elif "ID" not in df.columns:
        raise ValueError("Need an ID.")

    df.set_index(["ID", "timestamp", "dim"], inplace=True, drop=False)

    return df


def np_to_TSdf(arr, df=None, ID=None, timestamp=None, dim_label=None, feature="0"):
    """Convert a numpy array to pandas DataFrame."""
    # df
    if df is None:
        df = pd.DataFrame()
    else:
        df = df.copy()

    # ID
    if ID is None:
        raise ValueError("Need an ID.")

    # dim
    if dim_label is None:
        dim_label = ["0"]

    # Insert Feature into the DataFrame
    if arr.ndim == 3:
        for i in range(len(dim_label)):
            df[feature + dim_label[i]] = arr[:, :, i].flatten()
    elif arr.ndim == 2:
        df[feature + dim_label[0]] = arr.flatten()
    elif arr.ndim == 1:
        df[feature] = arr
    else:
        raise ValueError("Need a well-defined numpy array.")

    # Convert DataFrame to TimeSeries format
    df = df_to_TSdf(df, ID=ID, timestamp=timestamp, dim_label=dim_label)

    return df


def dict_to_TSdf(results, ID=None, timestamp=None, dim_label=None):
    """Convert a dict to pandas DataFrame."""
    df = pd.DataFrame()
    for feature in results:
        df = np_to_TSdf(
            results[feature],
            df,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature=feature,
        )
    return df
