"""DataFormat module.

These are example of how to format data.

"""

from __future__ import annotations
from TSbench.TSdata.TSloader import LoaderTSdf
import os
import pandas as pd
import shutil


def csv2pqt(
    path: str,
    filename: str,
    process_function: callable[[pd.DataFrame], None] = None,
    **loader_args: any,
) -> None:
    """Format a file from csv to pqt.

    Args:
        path (str): Path of the file.
        filename (str): Name of the file.
        process_function (callable[[pd.DataFrame]) : None] ) : Default is  None.
            A function to processs the data
        **loader_args (any): Keyword arguments for the loader.

    """
    datatype, ext = os.path.splitext(filename)
    if ext != ".csv":
        raise ValueError("`Filename` should be a csv file")

    loader = LoaderTSdf(path=path, datatype=datatype, **loader_args)
    df = pd.read_csv(os.path.join(path, filename))

    # process the data
    if process_function is not None:
        process_function(df)

    loader.add_datatype(df)
    loader.write()


def dataset_csv2pqt(
    path: str,
    process_function: callable[[pd.DataFrame], None] = None,
    **loader_args: any,
) -> None:
    """Format files in path from csv to pqt.

    Args:
        path (str): Path of the file.
        filename (str): Name of the file.
        process_function (callable[[pd.DataFrame]) : None] ) : Default is  None.
            A function to processs the data
        **loader_args (any): Keyword arguments for the loader.

    """
    for filename in os.listdir(path):
        _, ext = os.path.splitext(filename)
        if ext == ".csv":
            csv2pqt(path, filename, **loader_args)


def merge_dataset(
    loaders: list["LoaderTSdf"], merge_path: str, **merge_loader_args: any
) -> "LoaderTSdf":
    """Merge dataset assuming no shared dataype.

    The merge path needs to be distinct from the path of all loaders.

    Args:
        loaders (LoaderTSdf): List of loaders to merge data on.
        merge_path (str): List of loaders to merge data on.
        **merge_loader_args (any): Arguments for the outputed TSloader's constructor

    Returns:
        "LoaderTSdf": LoaderTSdf instance with the metadata attribute merged.

    Raises:
        ValueError: If `merge_path` is one of `loaders` path.

    """
    if type(loaders) is not list:
        raise ValueError("Give a list of the loaders to merge")

    merge_loader = LoaderTSdf(
        path=merge_path,
        datatype=loaders[0].datatype,
        permission="overwrite",
        **merge_loader_args,
    )

    i = 0
    for loader in loaders:
        if loader.path == merge_path:
            raise ValueError(
                "The merge path needs to be distinct " + "from the path of all loaders."
            )
    for filename in os.listdir(loader.path):
        if filename == "metadata.pqt":
            src = os.path.join(loader.path, filename)
            dst = os.path.join(merge_path, "metadata-" + str(i) + ".pqt")
            shutil.copyfile(src, dst)
            i += 1
        else:
            src = os.path.join(loader.path, filename)
            dst = os.path.join(merge_path, filename)
            shutil.copyfile(src, dst)

    merge_loader.merge_metadata(write_metadata=True, rm=True)
    return merge_loader
