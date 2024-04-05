### Format different data for TSdf.
import numpy as np
import pandas as pd

from TSbench.TSdata.DataFormat import (
    df_to_TSdf,
    dict_to_TSdf,
    list_np_to_TSdf,
    np_to_TSdf,
)
from TSbench.TSdata.TSloader import LoaderTSdf, convert_from_TSdf


def test_simple_types_data_format():
    path = "data/example_data_format/data"
    datatype = "DataToFormat"
    permission = "overwrite"  # Overwrite is used for repeated execution
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
    loader.restart_dataset()  # for fresh re-run

    ###  Format with custom timeseries with multiple features
    T = 10  # number of timestamp
    timestamp = list(pd.date_range(start="2021-01-01", periods=T))

    ### Value from DataFrame
    ID = "Conversion"
    d = {"feature0": np.arange(10), "feature1": np.arange(10, 20)}
    df = pd.DataFrame(data=d)
    TSdf1 = df_to_TSdf(df, ID=ID, timestamp=timestamp)
    loader.add_data(TSdf1, ID=ID, collision="overwrite")

    ### Value from np.ndarray
    arr_feature0 = np.arange(10)
    TSdf2 = np_to_TSdf(arr_feature0, ID=ID, timestamp=timestamp)
    loader.add_data(
        TSdf2, ID=ID, collision="overwrite", feature_label=np.array(["feature0"])
    )

    ### Value from a dictionary of multiple features
    arr_feature1 = np.arange(10)
    dict_features = {"feature0": arr_feature0, "feature1": arr_feature1}
    TSdf3 = dict_to_TSdf(dict_features, ID=ID, timestamp=timestamp)
    loader.add_data(TSdf3, ID=ID, collision="overwrite")

    ### Value from list of np.ndarray
    arr_feature = np.zeros((10, 3))
    mat_feature = np.ones((10, 3, 3))
    TSdf4 = list_np_to_TSdf(
        [arr_feature, mat_feature],
        ID=ID,
        timestamp=timestamp,
        dim_label=["First", "Second", "Third"],
    )
    loader.add_data(TSdf4, ID=ID, collision="overwrite")

    # dimension
    dim_label = ["FirstDimension"]

    ID1 = "FromDataFrame"
    ID2 = "FromDict"
    d = {"feature0": np.arange(10), "feature1": np.arange(10, 20)}
    df = pd.DataFrame(data=d)

    arr_feature0 = np.arange(10)
    arr_feature1 = np.arange(10, 20)
    dict_features = {"feature0": arr_feature0, "feature1": arr_feature1}

    TSdf1 = df_to_TSdf(df, ID=ID1, timestamp=timestamp, dim_label=dim_label)

    TSdf2 = dict_to_TSdf(
        dict_features, ID=ID2, timestamp=timestamp, dim_label=dim_label
    )
    # use add_data
    loader.add_data(ID=ID1, data=TSdf1, collision="overwrite")
    loader.add_data(ID=ID2, data=TSdf2, collision="overwrite")
    assert (
        loader.get_df(IDs=[ID1])
        .droplevel("ID")
        .equals(loader.get_df(IDs=[ID2]).droplevel("ID"))
    )

    loader.write()


def test_conversion_to_TSdf():
    ID = "Test"
    T = 10
    dim_label = ["0", "1"]
    # prepare solution
    d1 = {"feature": np.ones(10)}
    d2 = {"feature": np.ones(20)}
    d3 = {"feature0": np.ones(20), "feature1": np.ones(20)}
    sol1 = df_to_TSdf(pd.DataFrame(data=d1), ID=ID)
    sol2 = df_to_TSdf(pd.DataFrame(data=d2), ID=ID, dim_label=dim_label)
    sol3 = df_to_TSdf(pd.DataFrame(data=d3), ID=ID, dim_label=dim_label)

    # numpy dimension 1 (ndim)
    arr1 = np.ones((T))
    TSdf1 = np_to_TSdf(arr1, ID=ID)
    assert TSdf1.equals(sol1)
    # numpy imensions 2 (ndim)
    dim = 2
    arr2 = np.ones((T, dim))
    TSdf2 = np_to_TSdf(arr2, ID=ID, dim_label=dim_label)
    assert TSdf2.equals(sol2)
    # numpy dimensions 3 (ndim)
    nb_features = 2
    arr3 = np.ones((T, dim, nb_features))
    TSdf3 = np_to_TSdf(arr3, ID=ID, dim_label=dim_label)
    assert TSdf3.equals(sol3)


def test_conversion_from_TSdf():
    for N in range(1, 10):
        # numpy dimension 1 (ndim)
        N = 10
        ID = "Test"
        arr1 = np.ones(N)
        TSdf1 = np_to_TSdf(arr1, ID=ID)
        arr_copy1 = convert_from_TSdf(TSdf1, tstype=np.ndarray)
        assert arr1.shape == arr_copy1[:, 0].shape
        assert (arr1 == arr_copy1).all()

        # numpy dimension 2 (ndim)
        ID = "Test"
        arr2 = np.ones((N, N))
        TSdf2 = np_to_TSdf(arr2, ID=ID)
        arr_copy2 = convert_from_TSdf(TSdf2, tstype=np.ndarray)
        assert arr2.shape == arr_copy2.shape
        assert (arr2 == arr_copy2).all()

        # numpy dimension 3 (ndim)
        for nb_features in range(2, 11):
            ID = "Test"
            arr3 = np.ones((N, N, nb_features))
            TSdf3 = np_to_TSdf(arr3, ID=ID)
            arr_copy3 = convert_from_TSdf(TSdf3, tstype=np.ndarray)
            assert arr3.shape == arr_copy3.shape
            assert (arr3 == arr_copy3).all()
