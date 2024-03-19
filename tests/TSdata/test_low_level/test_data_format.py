### Format different data for TSdf.
import numpy as np
import pandas as pd
from TSbench.TSdata import LoaderTSdf, DataFormat


def test_data_format():
    path = "data/example_data_format/data"
    datatype = "DataToFormat"
    permission = "overwrite"  # Overwrite is used for repeated execution
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
    loader.restart_dataset()  # for fresh re-run

    ### Default value from DataFrame
    ID = "FromDataFrame"
    d = {"feature0": np.arange(10), "feature1": np.arange(10, 20)}
    df = pd.DataFrame(data=d)
    DataFormat.df_to_TSdf(df, ID=ID)
    loader.add_data(df, ID=ID, collision="overwrite")

    ### Default value from array
    arr_feature0 = np.arange(10)
    DataFormat.np_to_TSdf(arr_feature0, ID=ID)
    loader.add_data(df, ID=ID, collision="overwrite")

    ### Default value from a dictionary of multiple features
    arr_feature1 = np.arange(10)
    dict_features = {"feature0": arr_feature0, "feature1": arr_feature1}
    DataFormat.dict_to_TSdf(dict_features, ID=ID)
    loader.add_data(df, ID=ID, collision="overwrite")

    ###  Format with custom timeseries with multiple features
    T = 10  # number of timestamp
    timestamp = list(
        pd.date_range(start="2021-01-01", periods=T).strftime("%Y-%m-%d %X")
    )

    # dimension
    dim_label = ["FirstDimension"]

    ID = "FromDataFrame"
    d = {"feature0": np.arange(10), "feature1": np.arange(10, 20)}
    df = pd.DataFrame(data=d)

    arr_feature0 = np.arange(10)
    arr_feature1 = np.arange(10, 20)
    dict_features = {"feature0": arr_feature0, "feature1": arr_feature1}

    TSdf1 = DataFormat.df_to_TSdf(df, ID=ID, timestamp=timestamp, dim_label=dim_label)

    TSdf2 = DataFormat.dict_to_TSdf(
        dict_features, ID=ID, timestamp=timestamp, dim_label=dim_label
    )

    # Vizualize
    loader.add_data(ID=ID, data=TSdf1, collision="overwrite")

    # assert they are equal removing indexed columns for order problem
    assert loader.get_df().equals(TSdf2)

    loader.write()
