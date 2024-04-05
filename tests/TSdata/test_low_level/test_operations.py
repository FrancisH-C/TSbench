import os

import numpy as np
import pandas as pd

from TSbench.TSdata import DatasetOperations, LoaderTSdf


def test_operations():
    path = "data/example_operations/data"
    datatype = "simulated"
    permission = "overwrite"  # Overwrite is used for repeated execution
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
    loader.restart_dataset()  # fresh re-run

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": np.arange(10),
        "feature1": np.arange(10, 20),
        "feature2": np.arange(10, 20),
    }
    df = pd.DataFrame(data=d)
    loader.add_data(data=df)

    ID = "added_ID"
    d = {
        "timestamp": np.arange(0, 5),
        "feature1": np.arange(5),
        "feature2": np.arange(10, 15),
    }
    df = pd.DataFrame(data=d)
    loader.add_data(df, ID=ID, collision="overwrite")

    feature = "added_feature"
    d = {"timestamp": np.arange(10), feature: np.arange(10)}
    df = pd.DataFrame(data=d)
    loader.add_feature(df, ID="added_ID", feature=feature)

    loader.write()

    empty_loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)

    empty_loader.rm_datatype()
    assert empty_loader.df is not None and len(empty_loader.df) == 0

    loader.set_metadata(start=["2016-01-01"])
    # loader.append_to_metadata(start=["2016-01-01"])
    # loader.append_to_metadata(test=["0", "0"], test2=["1", "1"])

    loader.write()

    data_path = "data/example_operations/data"
    multiprocess_path = "data/example_multiprocessing"
    multiprocess_datatype = "splitted_data"
    copy_path = "data/example_operations/copy"
    move_path = "data/example_operations/move"
    merge_path = "data/example_operations/example_merge"
    permission = "overwrite"
    data_loader = LoaderTSdf(path=data_path, datatype=datatype, permission=permission)
    multiprocess_loader = LoaderTSdf(
        path=multiprocess_path, datatype=multiprocess_datatype, permission=permission
    )

    if os.path.exists(copy_path):
        os.rmdir(copy_path)
    data_loader.copy_dataset(copy_path)

    data_loader.move_dataset(move_path)

    data_loader.rm_dataset()
    data_loader.set_path(data_path)

    DatasetOperations.merge_dataset([data_loader, multiprocess_loader], merge_path)
