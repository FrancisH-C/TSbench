import pandas as pd
import numpy as np
from TSbench.TSdata import TSloader
import pytest


def simple_loader():
    """Simple loader for test."""
    path = "data/test_add"
    datatype = "simulated"
    permission = "overwrite"
    loader = TSloader(path, datatype, permission=permission)

    ID = "added_ID"
    feature = "added_feature"

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
    }
    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}
    df = pd.DataFrame(data=d)
    df_feature = pd.DataFrame(data=d_feature)

    loader.set_df(df=df.copy())
    df["ID"] = ID
    loader.add_ID(df.copy(), ID=ID, collision="overwrite")
    loader.add_feature(df_feature, ID=ID, feature=feature)

    return loader


IDs = ["name1", "added_ID"]
timestamps = ["0", "1", "5"]
dims = ["0"]


loader = simple_loader()

loader.get_df()
loader.get_df(IDs=IDs)
loader.get_df(timestamps=timestamps)
loader.get_df(dims=dims)
loader.get_df(IDs=IDs, timestamps=timestamps)
loader.get_df(timestamps=timestamps, dims=dims)
loader.get_df(IDs=IDs, timestamps=timestamps, dims=dims)
