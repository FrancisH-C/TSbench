import numpy as np
import pandas as pd
from TSbench.TSdata import LoaderTSdf, DataFormat


def test_metadata():
    path = "data/example_data_format/data"
    datatype = "DataToFormat"
    permission = "overwrite"  # Overwrite is used for repeated execution
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
    loader.restart_dataset()  # fresh re-run

    ID = "FromDataFrame"
    d = {"feature0": np.arange(10)}
    d = {"feature0": np.arange(10), "feature1": np.arange(10, 20)}
    df = pd.DataFrame(data=d)

    TSdf = DataFormat.df_to_TSdf(df, ID=ID)
    loader.add_data(df, ID=ID, collision="overwrite")

    loader.update_metadata_from_df()
