import logging

import numpy as np
import pandas as pd

from TSbench.TSdata import LoadersProcess, LoaderTSdf


def test_multiprocessing():
    datatype = "splitted_data"
    path = "data/example_multiprocessing"
    split_pattern = ["split0", "split1"]
    permission = "overwrite"  # Overwrite is used for repeated execution

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": np.arange(0, 10),
        "feature0": np.arange(10),
        "feature1": np.arange(10),
    }
    df1 = pd.DataFrame(data=d).drop("feature1", axis=1)
    df2 = df1.copy()
    df2 = pd.DataFrame(data=d).drop("feature0", axis=1)
    df2["timestamp"] = df2["timestamp"] + 10

    def loader_logger(loader):
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        logging.info(loader.df)

    # #### Set the splitting scheme
    metaloader = LoaderTSdf(
        path=path, datatype=datatype, split_pattern=split_pattern, permission=permission
    )
    metaloader.restart_dataset()  # fresh re-run
    metaloader.write_metadata()

    loader1 = LoaderTSdf(
        path=path,
        datatype=datatype,
        subsplit_pattern_index=[0],
        permission=permission,
        parallel=True,
    )
    loader2 = LoaderTSdf(
        path=path,
        datatype=datatype,
        subsplit_pattern_index=[1],
        permission=permission,
        parallel=True,
    )

    loader1.set_df(df1)
    loader2.set_df(df2)

    p = LoadersProcess(
        loaders=[loader1, loader2],
        loader_function=loader_logger,
        autoload=False,
        parallel=True,
    )
    p.run_loader()

    p = LoadersProcess(
        loaders=[loader1, loader2],
        loader_function=lambda loader: loader.write(),
        autoload=False,
        parallel=True,
    )
    p.run_loader()

    metaloader.merge_splitted_metadata(rm=False)

    metaloader = LoaderTSdf(path=path, datatype=datatype, permission=permission)

    loader1 = LoaderTSdf(
        path=path,
        datatype=datatype,
        subsplit_pattern_index=[0],
        permission=permission,
        parallel=True,
    )
    loader2 = LoaderTSdf(
        path=path,
        datatype=datatype,
        subsplit_pattern_index=[1],
        permission=permission,
        parallel=True,
    )

    p = LoadersProcess(
        loaders=[loader1, loader2],
        loader_function=loader_logger,
        autoload=True,
        parallel=True,
    )
    p.run_loader()
