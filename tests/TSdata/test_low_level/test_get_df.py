import numpy as np
import pandas as pd

from TSbench.TSdata import LoaderTSdf


def simple_loader():
    """Simple loader for test."""
    path = "data/test_add"
    datatype = "simulated"
    permission = "overwrite"
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)

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
    loader.add_data(df.copy(), ID=ID, collision="overwrite")
    loader.add_feature(df_feature, ID=ID, feature=feature)

    return loader


def test_get_df():
    IDs = np.array(["name1", "added_ID"])
    timestamps = np.array(["0", "1", "5"])
    dims = np.array(["0"])

    loader = simple_loader()
    loader.get_df()
    loader.get_df(timestamps=timestamps)
    loader.get_df(dims=dims)
    loader.get_df(IDs=IDs, timestamps=timestamps)
    loader.get_df(timestamps=timestamps, dims=dims)
    loader.get_df(IDs=IDs, timestamps=timestamps, dims=dims)
    loader.get_df(IDs=IDs)
    assert loader.get_df(IDs=IDs[0:1]).index.get_level_values("ID").unique() == IDs[0]
    assert (
        loader.get_df(timestamps=timestamps)
        .index.get_level_values("timestamp")
        .unique()
        == timestamps
    ).all()
    assert loader.get_df(dims=dims).index.get_level_values("dim").unique() == dims


def test_get_timestamp():
    timestamps = np.array(list(map(str, np.arange(10))))

    loader = simple_loader()
    assert (loader.get_timestamp() == np.repeat(np.array(timestamps), 2)).all()
    assert (loader.get_timestamp(unique=True) == timestamps).all()
    assert (loader.get_timestamp(start="2", unique=True) == timestamps[2:]).all()
    assert (loader.get_timestamp(end="2", unique=True) == timestamps[:2]).all()
    assert (
        loader.get_timestamp(start="1", end="2", unique=True) == timestamps[1:2]
    ).all()
