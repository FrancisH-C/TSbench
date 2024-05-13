import logging

import numpy as np
import pandas as pd

from TSbench.TSdata import LoadersProcess, LoaderTSdf


def test_multiprocessing_loaders():
    datatype = "splitted_data"
    data_path = "data/test_multiprocessing_loaders"
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
        logging.info(loader.df)

    # Set the splitting scheme
    metaloader = LoaderTSdf(
        path=data_path,
        datatype=datatype,
        split_pattern=split_pattern,
        permission=permission,
    )
    metaloader.restart_dataset()  # fresh re-run
    metaloader.write_metadata()

    loader1 = LoaderTSdf(
        path=data_path,
        datatype=datatype,
        subsplit_pattern_index=[0],
        permission=permission,
        autoload=False,
    )

    loader2 = LoaderTSdf(
        path=data_path,
        datatype=datatype,
        subsplit_pattern_index=[1],
        permission=permission,
    )

    loader1.set_df(df1)
    loader2.set_df(df2)

    p1 = LoadersProcess(
        input_loaders=[loader1, loader2],
        process_split=loader_logger,
        autoload=True,
    )

    p2 = LoadersProcess(
        input_loaders=[loader1, loader2],
        process_split=lambda loader: loader.write(),
        autoload=False,
    )

    p1.run_process()
    p2.run_process()


def test_multiprocessing_IDs():
    def half(df):
        """Calculate the half price of a given order.

        Write the result with output_loader

        Args:
            df (pd.DataFrame): Input to process.
            output_loader (LoaderTSdf): Used to write output.
        """
        return df / 2

    # Define dataset
    datatype = "splitted_data"
    path = "data/test_multiprocessing_IDs"
    split_pattern = ["split0"]
    permission = "overwrite"  # Overwrite is used for repeated execution
    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": np.arange(0, 10),
        "feature0": np.arange(10),
        "feature1": np.arange(10),
    }
    df = pd.DataFrame(data=d)

    dataset_loader = LoaderTSdf(
        path=path, datatype=datatype, split_pattern=split_pattern, permission=permission
    )
    dataset_loader.restart_dataset()  # fresh re-run
    dataset_loader.add_data(df)
    dataset_loader.write()

    # Process dataset
    half_process = LoadersProcess(
        data_path=path,
        datatype=datatype,
        n_input_loaders=1,
        n_jobs=2,
        process_df=half,
    )
    half_process.run_process()
