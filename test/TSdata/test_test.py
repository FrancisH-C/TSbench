import numpy as np
import pandas as pd
from TSbench.TSdata import TSloader, DataFormat


def test_low_level():
    path = "data/test_low_level"
    datatype = "simulated"
    loader = TSloader(path, datatype)

    ID = "ABC"
    # timeseries

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "dim": ["0" for _ in range(10)],
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
    }
    df = pd.DataFrame(data=d)

    loader.set_df(df=df)

    ID = "added_ID"
    d_ID = {
        "timestamp": list(map(str, range(0, 5))),
        "ID": [ID for _ in range(5)],
        "dim": ["0" for _ in range(5)],
        "feature0": list(range(5)),
        "feature1": list(range(10, 15)),
    }
    df_ID = pd.DataFrame(data=d_ID)

    loader.add_ID(df_ID, ID=ID, collision="overwrite")

    feature = "added_feature"
    d_feature = {
        "timestamp": list(map(str, range(0, 4))),
        "ID": [ID for _ in range(4)],
        "dim": ["0" for _ in range(4)],
        feature: list(range(10, 14)),
    }
    df_feature = pd.DataFrame(data=d_feature)

    loader.add_feature(df_feature, ID=ID, feature=feature)

    d_feature = {
        "timestamp": list(map(str, range(0, 6))),
        "ID": [ID for _ in range(6)],
        "dim": ["0" for _ in range(6)],
        feature: list(range(10, 16)),
    }
    df_feature = pd.DataFrame(data=d_feature)

    #loader.add_feature(df_feature, ID=ID, feature=feature)

    loader.write()
