"""Example of data format to a TimeSeries DataFrame."""

import pandas as pd
import numpy as np


def convert_to_TSdf(data, ID=None, timestamp=None, dim_label=None, feature_label=None):
    if data is None or len(data) == 0:
        return
    elif type(data) is pd.DataFrame:
        df = df_to_TSdf(
            data,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature_label=feature_label,
        )
    elif type(data) is list:
        # list of np.array
        if type(data[0]) is np.ndarray:
            df = list_np_to_TSdff(
                data,
                ID=ID,
                timestamp=timestamp,
                dim_label=dim_label,
                feature_label=feature_label,
            )
        #  list of list
        elif type(data[0]) is list:
            df = np_to_TSdf(
                np.array(data),
                ID=ID,
                timestamp=timestamp,
                dim_label=dim_label,
                feature_label=feature_label,
            )
        else:  # non-nested list
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
        # format timestamp to fit DataFrame dimension
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


def list_np_to_TSdff(
    arr_list, df=None, ID=None, timestamp=None, dim_label=None, feature_label=None
):

    # ID
    if ID is None:
        raise ValueError("Need an ID.")

    # df
    if df is None:
        df = pd.DataFrame()
    else:
        df = df.copy()

    # dim
    if dim_label is None:
        if arr_list[0].ndim < 2:
            dim_label = ["0"]
        else:
            dim_label = [i for i in range(arr_list[0].shape[1])]

    # default feature_label
    if feature_label is None:
        feature_label = []
        feature_id = 0
        for arr in arr_list:
            if arr.ndim < 3:
                features = ["feature" + str(feature_id)]
                feature_id += 1
            else:
                features = [
                    "feature" + str(feature_id + i) for i in range(arr.shape[2])
                ]
                feature_id += arr.shape[2]

            feature_label.extend(features)

    if len(feature_label) < len(arr_list):
        raise ValueError(
            "You need to have, at least, the same number of feature_label"
            + " than the length of arr_list."
        )

    # map feature_label to element in the list
    feature_counter = 0
    for arr in arr_list:
        if arr.ndim < 3:
            current_features = feature_label[feature_counter]
            feature_counter += 1
        else:
            current_features = feature_label[feature_counter : arr.shape[2] + 1]
            feature_counter += arr.shape[2]
        df[current_features] = np_to_TSdf(
            arr,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature_label=current_features,
        )

    # Convert DataFrame to TSdf format
    df = df_to_TSdf(df, ID=ID, timestamp=timestamp, dim_label=dim_label)
    return df


def np_to_TSdf(
    arr, df=None, ID=None, timestamp=None, dim_label=None, feature_label=None
):
    # ID
    if ID is None:
        raise ValueError("Need an ID.")

    # df
    if df is None:
        df = pd.DataFrame()
    else:
        df = df.copy()

    # dim
    if dim_label is None:
        if arr.ndim < 2:
            dim_label = ["0"]
        else:
            dim_label = [i for i in range(arr.shape[1])]

    if feature_label is None:
        if arr.ndim < 3:
            feature_label = ["feature"]
        else:
            feature_label = ["feature" + str(i) for i in range(arr.shape[2])]

    # Insert feature_label into the DataFrame
    # ndim 1
    if arr.ndim == 1:
        df[feature_label[0]] = arr
    elif arr.shape[1] == len(dim_label):
        if arr.ndim == 2:  # ndim 2
            df[feature_label[0]] = arr.flatten()
        elif arr.ndim == 3 and arr.shape[2] == len(feature_label):  # ndim 3
            for i in range(len(feature_label)):
                df[feature_label[i]] = arr[:, :, i].flatten()
        else:
            raise ValueError(
                f"'arr' along axis 2 is {arr.shape[2]} and 'feature_label' "
                + f"is {len(feature_label)}. Need the same dimension."
            )
    else:
        raise ValueError(
            f"'arr' along axis 1 is {arr.shape[1]} and 'dim_label' "
            + f"is {len(dim_label)}. Need the same dimension."
        )

    # Convert DataFrame to TSdf format
    df = df_to_TSdf(df, ID=ID, timestamp=timestamp, dim_label=dim_label)
    return df


def dict_to_TSdf(data, ID=None, timestamp=None, dim_label=None):
    """Convert a dict to pandas DataFrame."""
    df = pd.DataFrame()
    for feature in data:
        df = np_to_TSdf(
            data[feature],
            df,
            ID=ID,
            timestamp=timestamp,
            dim_label=dim_label,
            feature_label=[feature],
        )
    return df
