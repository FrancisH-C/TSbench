"""Operations on dataset to transform data set.

- csv -> pqt
- pqt -> csv
- Merge multiple datasets into one

"""

from __future__ import annotations

import os
import shutil
from typing import Any

from TSbench.TSdata.TSloader import LoaderTSdf


def merge_dataset(
    loaders: list[LoaderTSdf], merge_path: str, **merge_loader_args: Any
) -> LoaderTSdf:
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
    if not isinstance(loaders, list):
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

    merge_loader.merge_splitted_metadata(write_metadata=True, rm=True)
    return merge_loader
