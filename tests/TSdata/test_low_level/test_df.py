import numpy as np
import pandas as pd
from TSbench.TSdata import LoaderTSdf


def test_low_level():
    path = "data/test/"
    datatype = "simulated"
    permission = "overwrite"  # Overwrite is used for repeated execution
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
    loader.restart_dataset()  # fresh re-run

    ### Simple usage
    path = "data/quickstart/"
    datatype = "Stock"
    loader = LoaderTSdf(path=path, datatype=datatype)

    ### Add data
    d_ID = {"feature0": np.arange(10), "feature1": np.arange(10, 20)}

    df_ID = pd.DataFrame(data=d_ID)

    ### add ID
    loader.df = pd.DataFrame()
    ID = "added_ID"
    loader.add_data(df_ID, ID=ID, collision="overwrite")
    loader.df  # in memory

    ### add feature
    ID = "added_ID"
    feature = "added_feature"
    d_feature = {"timestamp": np.arange(4), feature: np.arange(10, 14)}
    df_feature = pd.DataFrame(data=d_feature)

    loader.add_feature(df_feature, ID=ID, feature=feature)

    ### override df
    d_dtype = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "dim": ["0" for _ in range(10)],
        "timestamp": np.arange(0, 10),
        "feature0": np.arange(10),
        "feature1": list(range(10, 20)),
    }

    df_dtype = pd.DataFrame(data=d_dtype)

    loader.set_df(df=df_dtype)
