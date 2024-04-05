import numpy as np
import pandas as pd

from TSbench.TSdata import DataFormat, LoaderTSdf


def test_multivariate():
    """An multivariate test example with 2 dimensions.

    The features are 'returns' and 'vol' for a 'SimulatedStock' datatype."""
    path = "data/example_multivariate/data"
    datatype = "SimulatedStock"
    permission = "overwrite"  # Overwrite is used for repeated execution
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
    loader.restart_dataset()  # fresh re-run

    ID1 = "ABC"
    ID2 = "XYZ"
    dim_label = np.array(["0", "1"])
    returns = np.array([[0, 1], [2, 3], [4, 5]])
    vol = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])

    df1 = pd.DataFrame({"returns": returns.flatten()})
    df2 = pd.DataFrame({"returns": returns.flatten()})
    df2["vol0"] = vol[:, :, 0].flatten()
    df2["vol1"] = vol[:, :, 1].flatten()

    loader.add_data(df1, ID=ID1, dim_label=dim_label, collision="overwrite")

    loader.add_data(df2, ID=ID2, dim_label=dim_label, collision="overwrite")

    ID1 = "ABC"
    ID2 = "XYZ"
    # timeseries
    returns = np.array([[0, 1], [2, 3], [4, 5]])
    vol = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])

    df1 = DataFormat.np_to_TSdf(
        returns, ID=ID1, dim_label=dim_label, feature_label=["returns"]
    )
    df1 = DataFormat.np_to_TSdf(
        vol, df=df1, ID=ID1, dim_label=dim_label, feature_label=["vol0", "vol1"]
    )
    d = {"returns": returns, "vol0": vol[:, :, 0], "vol1": vol[:, :, 1]}
    df2 = DataFormat.dict_to_TSdf(d, ID=ID1, dim_label=dim_label)
    assert df1.equals(df2)
