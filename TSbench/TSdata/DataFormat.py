"""Example of data format to a TimeSeries DataFrame."""
import pandas as pd
import numpy as np


def convert_to_TSdf(data, ID=None, timestamp=None, dim_label=None, feature_label=None):
    if data is None or len(data) == 0:
        return
    if type(data) is pd.DataFrame:
        df = df_to_TSdf(
            data,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature_label=feature_label,
        )
    elif type(data) is list:
        # list of arrays or list of list
        if type(data[0]) is np.ndarray or type(data[0]) is list:
            df = listnp_to_TSdf(
                data,
                ID=ID,
                timestamp=timestamp,
                dim_label=dim_label,
                feature_label=feature_label,
            )
        else: # non-nested list
            df = np_to_TSdf(
                np.array(data),
                ID=ID,
                timestamp=timestamp,
                dim_label=dim_label,
                feature_label=feature_label,
            )
    elif type(data) is np.ndarray:
        df = np_to_TSdf(
            data,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature_label=feature_label,
        )
    elif type(data) is dict:
        df = dict_to_TSdf(
            data,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature_label=feature_label,
        )
    else:
        raise ValueError("Data is of the wrong type to format")
    return df


def df_to_TSdf(df, ID=None, timestamp=None, dim_label=None, feature_label=None):
    """Convert a pandas DataFrame into a TimeSeries DataFrame.

    By default, keeps data in the columns of the DataFrame.
    ID is overridden
    feature is overridden

    """
    # dim
    df = df.copy()
    df = df.reset_index()  # put all data in columns
    if "index" in list(df.columns):  # remove index column is it's there
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
            timestamp = list(range(0, T))
            timestamp = np.transpose(
                np.array([timestamp for _ in range(len(dim_label))])
            ).flatten()
        df["timestamp"] = timestamp

    # ID
    if ID is not None:
        df["ID"] = ID
        if "ID" not in df.columns:
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

    if feature_label is not None:
        df.columns = feature_label

    return df


def listnp_to_TSdf(
    arr_list, df=None, ID=None, timestamp=None, dim_label=None, feature_label=None
):
    # df
    if df is None:
        df = pd.DataFrame()
    else:
        df = df.copy()

    # dim
    if dim_label is None:
        dim_label = ["0"]

    if feature_label is None:
        feature_label = ["feature"]

    # ID
    if ID is None:
        raise ValueError("Need an ID.")

    if len(feature_label) != len(arr_list):
        raise ValueError("feature_label needs the same length as arr_list.")

    for i in range(len(feature_label)):
        df[feature_label[i]] = arr_list[i].flatten()

    # Convert DataFrame to TSdf format
    df = df_to_TSdf(df, ID=ID, timestamp=timestamp, dim_label=dim_label)
    return df


def np_to_TSdf(
    arr, df=None, ID=None, timestamp=None, dim_label=None, feature_label=None
):
    # df
    if df is None:
        df = pd.DataFrame()
    else:
        df = df.copy()

    # dim
    if dim_label is None:
        dim_label = ["0"]

    if feature_label is None:
        feature_label = ["feature"]

    # ID
    if ID is None:
        raise ValueError("Need an ID.")

    # Insert feature_label into the DataFrame
    if arr.ndim == 3 and len(dim_label) == arr.shape[-1]:
        for i in range(len(dim_label)):
            df[feature_label[i]] = arr[:, :, i].flatten()
    elif arr.ndim == 2 and len(dim_label) == arr.shape[-1]:
        df[feature_label[0]] = arr.flatten()
    elif arr.ndim == 1:
        df[feature_label[0]] = arr
    else:
        raise ValueError(
            f"'arr' along axis 1 is {arr.shape[-1]} and 'dim_label' "
            + f"is {len(dim_label)}. Need the same dimension."
        )

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
            feature_label=[feature],
        )
    return df
