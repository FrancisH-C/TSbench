"""Example of data format to a TimeSeries DataFrame."""
import pandas as pd
import numpy as np

def df_to_TSdf(df, ID=None, timestamp=None, dim_label=None):
    """Convert a pandas DataFrame into a TimeSeries DataFrame.

    By default, keeps overwrite with data in the columns of the DataFrame.

    """
    # dim
    df = df.copy()
    #df["dim"] = [0,0,0,0,0]
    df = df.reset_index() # put all data in columns
    if "index" in list(df.columns): # remove index column is it's there
        df = df.drop(columns=["index"])


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
    if "ID" not in df.columns:
        if ID is not None:
            df["ID"] = ID
        else:
            raise ValueError("Need an ID.")

    # if in columns, put it in index
    index = []
    if "ID" in list(df.columns):
        index.append("ID")
    if "timestamp" in list(df.columns):
        index.append("timestamp")
    if "dim" in list(df.columns):
        index.append("dim")

    if len(index) > 0:
        df.set_index(index, inplace=True)

    return df


def np_to_TSdf(arr, df=None, ID=None, timestamp=None, dim_label=None, feature="0"):
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

    # Convert DataFrame to TSdf format
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
