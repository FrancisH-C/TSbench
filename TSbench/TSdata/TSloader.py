from __future__ import annotations
from abc import ABC
import pandas as pd
import numpy as np
import math
import os
import shutil
import logging
from typing import Callable

from TSbench.TSdata.DataFormat import convert_to_TSdf
from joblib import Parallel, delayed


class TSloader(ABC):
    def __init__(
        self,
        path: str = "data/",
        datatype: str = None,
        split_pattern: np.ndarray = np.array([]),
        subsplit_pattern: np.ndarray = np.array([]),
        subsplit_pattern_index: np.ndarray = np.array([]),
        parallel: bool = False,
        permission: str = "overwrite",
        autoload: bool = True,
    ) -> "TSloader":
        """Init method."""
        # Permissions
        self.set_permission(permission)  # read, write, overwrite

        # Initialize without parallel
        self.parallel = False

        # Set path
        self.set_path(path)

        # Initialize metadata
        self.load_metadata()

        ## Set the datatype
        self.set_datatype(
            datatype, split_pattern, subsplit_pattern, subsplit_pattern_index
        )
        # Initialize df
        if autoload:
            self.df = self.load()

        # For parallel usage
        self.parallel = parallel

    ######################
    # dataset operations #
    ######################

    def set_path(self, path: str, ensure_path=True) -> None:
        """Set the current path.

        Args:
            path (str): The path to set.
        """
        self.path = path
        if ensure_path:
            self._create_path()


class LoaderTSdf(TSloader):
    """Use to write, load and modify a timeseries dataset.

    A TSloader is assigned a path to a "dataset". Optionally, it can have a
    "datatype" which informs about the structure of the data. "Datatype" is a
    collection of multiple input with different "IDs". A given "datatype" as the
    exact same "features" which is the data indexed with a "timestamp" (the
    timeseries).

    A "datatype" can be splitted on different files on disk, this is called a
    "split_pattern". A TSloader with that "datatype" and a "subsplit" (either with names
    or indices) can manipulate the data from the files. It is used when a single
    datatype is too large or for parallelization purposes.

    The split_pattern is an "attribute" of a the datatype, stored in metadata. It can be updated if new
    data of split patterns are needed, but it is more fixed as a way to name the file
    for the data. It is stored in memory as metadata.

    On the other hand, the subsplit_pattern, is more dynamic, it depends on the
    specific TSloader used to load the data. It tells the TSloader the (sub)-files to
    load for a specific datatype. For an example see "example_multiprocess.ipynb".

    Notes :
        Most of the attributes are better changed using their 'set' method or by
        pdefining a new loader.

    Args:
        path (str): Sets attribute of the same name.
        datatype (str): Sets attribute of the same name.
        split_pattern (np.ndarray, optional): Sets attribute of the same name.
        subsplit_pattern (np.ndarray , optional): The subsplit scheme to use.
            Default is to use the whole split.
        subsplit_pattern_index (np.ndarray , optional): The indices to use in subsplit.
            Default is to use all the indices from the split.
        parallel (bool, optional): Sets attribute of the same name.
        permission (str, optional): Sets attribute of the same name.

    Attributes:
        path (str): The path to the dataset.
        datatype (str): The type of data which inform about the
            structure of the data. It is used as part of the file name in the
            dataset.
        df (pd.DataFrame): The pandas' dataset.
        metadata (pd.DataFrame): The pandas' metadata.
        split_pattern (np.ndarray, optional): A given datatype is store in a sequence of
            splits. Used when a single datatype is too large or for
            parallelization.
        subsplit_pattern (np.ndarray , optional): The subsplit scheme to use.
            Default is to use the whole split_pattern.
        subsplit_pattern_index (np.narray , optional): The indices to use as subsplit_pattern.
            Default is to use all the indices from the split.
        parallel (bool): Parallel informn on how to manipulate metadata.
            Parallel must be set to True to use in parallel to be used in
            parallel. Default is False.
        permission (str): To choose between {'read', 'write', 'overwerite'},
            with an increasing level of permission for the loader. The options are :

            Permission is seen as permission for operations on disk.
            - 'read' : Read only the data on disk and change it on memory.
            - 'write' : Add only new datatype and new ID. Any operation that \
               would remove or change data or metadata will raise an error.
            - 'overwrite' (default) :  Do all operations and write on disk.

    """

    def __init__(self, **TSloader_args) -> "TSloader":
        """Init method."""
        # Permissions
        super().__init__(**TSloader_args)

    ######################
    # dataset operations #
    ######################

    def set_path(self, path: str, ensure_path=True) -> None:
        """Set the current path.

        Args:
            path (str): The path to set.
        """
        self.path = path
        if ensure_path:
            self._create_path()

    def _create_path(self) -> None:
        """Create the dataset if it doesn't exsist."""
        # if path doesn't exsist.
        if self.path != "" and not os.path.isdir(self.path):
            # Create path
            if self.permission != "read":
                logging.info(f"Path '{self.path}' does not exist, creating.")
                os.makedirs(self.path)
            else:
                raise ValueError(
                    "To create the path, you need more than the 'read' permission"
                )

    def _append_path(self, filename: str) -> str:
        """Give the filename appended with the path attribute.

        Args:
            filename (str): Name of the file.

        Returns:
            str: Filename with appended the loader path.

        """
        return os.path.join(self.path, filename)

    def set_permission(self, permission="write") -> None:
        """Set the current path.

        Args:
            permission (str, optional): To choose between {'read', 'write',
                'overwerite'}, with an increasing level of permission for the loader.
                The options are :

                - 'read' : Read only the data on disk and change it on memory.
                - 'write' : Add only new datatype and new ID. Any operation that \
                   would remove or change data or metadata will raise an error.
                - 'overwrite' (default) :  Do all operations and write on disk.

        """
        if permission not in ["read", "write", "overwrite"]:
            raise ValueError("Permission is either 'read', 'write' or 'overwrite'")

        self.permission = permission

    def rm_dataset(self, ignore_errors=True) -> None:
        """Remove dataset. Dangerous method.

        Args:
            ignore_errors (str, optional): If ignore_errors is set to True, errors
                    arising due to unsuccessful file removals will be ignored. This
                    is set to `True` by default

        Raises:
            ValueError: If permission is not overwerite.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove the dataset, you need 'overwrite' permission")

        shutil.rmtree(self.path, ignore_errors=ignore_errors)

    def restart_dataset(self):
        self.rm_dataset(ignore_errors=True)
        self._create_path()

    def move_dataset(self, new_path: str) -> None:
        """Move dataset to another location.

        Change the path.

        Args:
            new_path (str): The path where to move the data.

        Raises:
            ValueError: If permission is not ovwerwrite or `self.path` is
                equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """

        if self.permission != "overwrite":
            raise ValueError("To move the dataset, you need 'overwrite' permission")
        if os.path.isdir(new_path):
            raise OSError(
                f"'{new_path}' already exists, "
                + "to merge dataset use `merge_dataset`."
            )

        old_path = self.path
        self.set_path(new_path)

        for basename in os.listdir(old_path):
            filename = os.path.join(old_path, basename)
            if os.path.isfile(filename):
                os.rename(filename, self._append_path(basename))

    def copy_dataset(self, new_path: str) -> None:
        """Copy dataset to another location.

        Change the path.

        Args:
            new_path (str): The path where to copy the data.

        Raises:
            ValueError: If `self.path` is equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """
        if self.permission == "read":
            raise ValueError("To copy the dataset, you need 'write' permission")

            raise ValueError(f"'{new_path}' is already the current dataset path.")
        if os.path.isdir(new_path):
            raise OSError(
                f"'{new_path}' already exists, "
                + "to merge dataset use `merge_dataset`."
            )

        old_path = self.path
        self.set_path(new_path)

        for basename in os.listdir(old_path):
            filename = os.path.join(old_path, basename)
            if os.path.isfile(filename):
                shutil.copyfile(filename, self._append_path(basename))

    def copy_datatype(self, new_path: str) -> None:
        """Copy dataset to another location.

        Args:
            new_path (str): The path where to copy the data.

        Raises:
            ValueError: If `self.path` is equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """
        if self.permission == "read":
            raise ValueError("To copy the dataset, you need 'write' permission")
        if self.path == new_path:
            raise ValueError(f"'{new_path}' is already the current dataset path.")

        metadata_file = self.get_filename(self, for_metadata=True)

        old_path = self.path
        # create new path
        self.set_path(new_path)

        new_metadata_file = self.get_filename(self, for_metadata=True)
        shutil.copyfile(metadata_file, new_metadata_file)

        for split in self.get_split_pattern():
            self.set_current_split(split, autoload=False)
            dst = self.get_filename()
            src = os.path.join(old_path, os.path.basename(dst))
            shutil.copyfile(src, dst)

    def list_datatypes(self) -> np.ndarray:
        """List datatypes from dataset."""
        return np.unique(np.array(self.metadata.index))

    #######################
    # metadata operations #
    #######################

    def _add_datatype_to_metadata(self) -> None:
        """Add the current datatype to the metadata indices."""
        if self.metadata.empty:
            self.metadata = pd.DataFrame(
                columns=["datatype", "IDs", "features", "split_pattern"]
            )
            self.metadata["datatype"] = [self.datatype]
            self.metadata.set_index(["datatype"], inplace=True, drop=True)

            self.metadata.at[self.datatype, "IDs"] = np.array([])
            self.metadata.at[self.datatype, "features"] = np.array([])
            self.metadata.at[self.datatype, "split_pattern"] = np.array([])
        elif self.datatype not in self.metadata.index:
            datatype = self.metadata.index.append(pd.Index([self.datatype]))
            self.metadata = self.metadata.reindex(datatype, fill_value=np.array([[]]))
        # else datatype is already in metadata indices

    def _update_split_pattern_to_metadata(
        self, split_pattern: np.ndarray = np.array([])
    ) -> None:
        """Set split pattern in "metadata"

        Update split_pattern in metadata if needed, otherwise do nothing.

        Args:
            split_pattern (np.ndarray, optional):

        Raises:
            ValueError: If split_pattern exists and no overwrite permission is granted.

        """
        # if split_pattern is empty
        if np.size(split_pattern) == 0:
            if np.size(self.metadata.at[self.datatype, "split_pattern"]) != 0:
                # split pattern already defined in metadata, do nothing
                return
            split_pattern = np.array([])
        self.set_metadata(split_pattern=split_pattern)

    def set_metadata(self, **metadata: np.ndarray):
        """set metadata.

        Args:
            **metadata (np.ndarray):

        Raises:
            ValueError: If permission is not overwrite.

        """
        for key in metadata:
            if type(metadata[key]) is not np.ndarray:
                metadata[key] = np.array(metadata[key], ndmin=1)
            if key not in self.metadata.columns:
                self.metadata[key] = pd.Series(data=np.array([]))
            elif np.size(self.metadata[key]) > 0 and self.permission != "overwrite":
                raise ValueError(
                    f"{key} already exsists in metadata. "
                    + "To overwite it, you need the overwrite permission."
                )
            self.metadata.at[self.datatype, key] = metadata[key]

            # if ndim and length is 1, pandas reduce it to dim to 0.
            # This brings it back as it is suppose to be: ndim == 1.
            if metadata[key].ndim == 1 and len(metadata[key]) == 1:
                self.metadata = self.metadata.map(lambda arr: np.array(arr, ndmin=1))

    def update_metadata_from_df(self, df=None) -> None:
        """Update metadata using df."""
        if df is None:
            df = self.df
        self.append_to_metadata(
            IDs=np.unique(np.array(df.index.get_level_values("ID")))
        )
        self.append_to_metadata(features=np.array(df.columns))

    def update_split_from_dataset(self) -> None:
        """Update metadata using df."""
        split_pattern = []
        for filename in os.listdir(self.path):
            if filename[0 : len(self.datatype)] == self.datatype:
                filename = os.path.splitext(filename)[0]
                split_pattern.append(filename[len(self.datatype) + 1 :])

        split_pattern = np.sort(split_pattern)
        self.set_metadata(split_pattern=split_pattern)

    def update_dataset_metadata(self) -> None:
        """Update metadata using df."""
        for datatype in self.list_datatypes():
            self.set_datatype(datatype)
            self.df = self.load()

            self.update_metadata_from_df()
            self.update_split_from_dataset()

        if not self.metadata.empty:
            self.write_metadata()

    def append_to_metadata(self, **metadata: np.ndarray) -> None:
        """Verify if entry is already there before append.

        Args:
            **metadata (np.ndarray):
        """
        for key in metadata:
            if key not in self.metadata.columns:
                self.metadata[key] = ""
                self.metadata.at[self.datatype, key] = metadata[key]

            updated_metadata = np.unique(
                np.append(self.metadata.at[self.datatype, key], [metadata[key]])
            )
            self.set_metadata(**{key: updated_metadata})

    def merge_metadata(self, write_metadata: bool = True, rm: bool = True) -> None:
        """Merge metadata between 'metadata-' file.

        Args:
            write (bool, optional): Whether or not to write the merged metadata.
            rm (bool, optional): Whether or not to removed all the 'metadata-' files
                after the merge.

        Raises:
            ValueError: If trying to write metadata with 'read'
                permission; Or if trying to remove metadata (on disk or
                memory) without 'overwrite' permission; Or if `parallel`
                attribute is `True`.

        """
        if self.permission == "read" and self.write:
            raise ValueError(
                "You cannot write metadata while merging " + "with 'read' permission."
            )
        elif self.permission != "overwrite" and rm:
            raise ValueError(
                "You cannot remove metadata while merging "
                + "without 'overwrite' permission."
            )

        elif not self.metadata.empty and self.permission != "overwrite":
            raise ValueError(
                "Trying to merge metadata but it already exists. "
                + "To force it, change permission to 'overwrite'."
            )

        if self.parallel:
            raise ValueError(
                "Set the parallel execution attribute " + "to `False` before merging."
            )

        datatype_keep = self.datatype
        self.metadata = pd.DataFrame()
        for filename in os.listdir(self.path):
            if filename[0:9] == "metadata-":
                metadata_file = self._append_path(filename)
                new_metadata = pd.read_parquet(metadata_file)
                for datatype in new_metadata.index:
                    self.datatype = datatype
                    self._add_datatype_to_metadata()
                    features = new_metadata.loc[self.datatype, "features"]
                    IDs = new_metadata.loc[self.datatype, "IDs"]
                    split_pattern = new_metadata.loc[self.datatype, "split_pattern"]

                    self.append_to_metadata(
                        split_pattern=split_pattern, IDs=IDs, features=features
                    )

                # remove metadata-* file
                if rm:
                    os.remove(metadata_file)

        self.datatype = datatype_keep
        if write_metadata and not self.metadata.empty:
            self.write_metadata()

    @staticmethod
    def merge_splitted_files(
        loader, n_jobs, write_metadata: bool = True, rm: bool = True
    ) -> None:
        if loader.permission == "read" and self.write:
            raise ValueError(
                "You cannot write metadata while merging " + "with 'read' permission."
            )
        elif loader.permission != "overwrite" and rm:
            raise ValueError(
                "You cannot remove metadata while merging "
                + "without 'overwrite' permission."
            )
        elif not loader.metadata.empty and loader.permission != "overwrite":
            raise ValueError(
                "Trying to merge metadata but it already exists. "
                + "To force it, change permission to 'overwrite'."
            )
        if loader.parallel:
            raise ValueError(
                "Set the parallel execution attribute " + "to `False` before merging."
            )

        def merge_day_wise(loader, new_split):
            merge_loader = LoaderTSdf(path=loader.path, datatype=loader.datatype)
            merge_loader.set_metadata(split_pattern=new_split_pattern)
            merge_loader.set_current_split(new_split)
            for old_split in loader.get_split_pattern():
                if old_split[split_substr_start:split_substr_end] == new_split:
                    loader.set_current_split(old_split)
                    merge_loader.add_data(
                        loader.df, format_df=False, update_metadata=False
                    )
                    if rm:
                        os.remove(loader.get_filename())
            merge_loader.write(write_metadata=False)

        # change metadata split_pattern
        split_substr_start = 0
        split_substr_end = 8
        new_split_pattern = np.unique(
            np.array(
                list(
                    map(
                        lambda split: split[split_substr_start:split_substr_end],
                        loader.get_split_pattern(),
                    )
                )
            )
        )

        Parallel(n_jobs=n_jobs)(
            delayed(merge_day_wise)(loader, new_split)
            for new_split in new_split_pattern
        )

        if write_metadata:
            metadata_loader = LoaderTSdf(path=loader.path, datatype=loader.datatype)
            metadata_loader.set_metadata(split_pattern=new_split_pattern)
            metadata_loader.write_metadata()

    def toggle_parallel(self) -> None:
        """Toggle parallel option."""
        if self.parallel:
            self.parallel = False
            print("Parallel mode deactivated")
        else:
            self.parallel = True
            print("Parallel mode activated")

    def load_metadata(self) -> pd.DataFrame:
        """Load datataset's metadata.

        Returns:
            pd.DataFrame: The pandas' metadata.

        """
        metadata_file = self.get_filename(for_metadata=True)
        if os.path.isfile(metadata_file):
            self.metadata = pd.read_parquet(metadata_file)
        else:
            self.metadata = pd.DataFrame()

        return self.metadata

    def write_metadata(self) -> None:
        """Write datataset's metadata.

        Raises:
            ValueError: If permission is read.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        self.metadata.to_parquet(self.get_filename(for_metadata=True))

    #######################
    # datatype operations #
    #######################

    def set_datatype(
        self,
        datatype: str,
        split_pattern: np.ndarray = np.array([]),
        subsplit_pattern: np.ndarray = np.array([]),
        subsplit_pattern_index: np.ndarray = np.array([]),
    ) -> None:
        """Change datatype and split_pattern used to load data.

        Args:
            datatype (str): The datatype to set.
            split_pattern (np.ndarray, optional): The split_pattern to set.
            subsplit_pattern (np.ndarray, optional): The subsplit names to set.
            subsplit_pattern_index (np.ndarray, optional): The subsplit indices to set.

        Raises:
            ValueError: If `datatype` is undefined.

        """
        if datatype is None:
            return

        self.datatype = datatype

        # update metadata
        self._add_datatype_to_metadata()
        self._update_split_pattern_to_metadata(split_pattern)

        # set subsplit_pattern for loader
        self.set_subsplit_pattern(subsplit_pattern, subsplit_pattern_index)

    def train_test_split(self, train_size=None, test_size=None, rounding="before"):
        if train_size is None and test_size is None:
            train_size = 0.7
            test_size = 0.3
        elif train_size is None:
            train_size = 1 - test_size
        elif test_size is None:
            test_size = 1 - train_size

        if train_size + test_size != 1:
            raise ValueError("The sum of 'train_size' and 'test_size' must be 1.")

        split_index = train_size * self.get_df().shape[0]
        if rounding == "before":
            split_index = math.floor(split_index)
        elif rounding == "after":
            split_index = math.ceil(split_index)

        train = self.get_df(end_index=split_index)
        test = self.get_df(start_index=split_index + 1)
        return train, test

    def get_df(
        self,
        start=None,
        start_index=None,
        end=None,
        end_index=None,
        IDs=None,
        timestamps=None,
        dims=None,
        features=None,
    ):
        """Alias for get_timeseries

        Use get_timeseries instead. About to be deprecated.
        """
        return self.get_timeseries(
            start=start,
            start_index=start_index,
            end=end,
            end_index=end_index,
            IDs=IDs,
            timestamps=timestamps,
            dims=dims,
            features=features,
        )

    def get_timestamp(
        self, start=None, start_index=None, end=None, end_index=None, IDs=None
    ):
        """Get timestamp for the datatype."""
        if IDs is None:
            IDs = slice(None)

        if start is not None and start_index is not None:
            raise ValueError("Either give start or a start_index")
        elif end is not None and end_index is not None:
            raise ValueError("Either give end or a end_index")

        # get all timestamps
        timestamps = pd.unique(self.df.loc[IDs].index.get_level_values("timestamp"))
        # If len(IDs) == 1, timestamps is sorted.
        if type(IDs) is slice or len(IDs) != 1:
            timestamps = np.sort(timestamps)
        if start is not None:
            start_index = timestamps.searchsorted(start)
        if end is not None:
            end_index = timestamps.searchsorted(end)

        return timestamps[start_index:end_index]

    def get_timeseries(
        self,
        start=None,
        start_index=None,
        end=None,
        end_index=None,
        IDs=None,
        timestamps=None,
        dims=None,
        features=None,
    ):
        """Get DataFrame for the datatype.

        Much more efficient if IDs is provided and list is short

        If "IDs", "timestamps" or "dims" is specify, fix that value. Otherwise
        returns the entries corrresponding to all values. The more efficient in
        memory is to have, at most, one specific fix value.
        """
        args = [start, start_index, end, end_index, IDs, timestamps, dims, features]
        if self.df.empty or all(arg is None for arg in args):
            return self.df

        # timestamps related args are not all None
        if timestamps is None and any(arg is not None for arg in args[0:4]):
            timestamps = self.get_timestamp(
                start=start,
                start_index=start_index,
                end=end,
                end_index=end_index,
                IDs=IDs,
            )
        else:
            timestamps = slice(None)

        # What follows does self.df.loc[IDs, timestamps, dims][features], but much more quickly
        if IDs is None:
            IDs = slice(None)
        if dims is None:
            dims = slice(None)
        if features is None:
            features = slice(None)

        df = self.df.loc[IDs]

        # keep index information in columns
        df = df.reset_index(drop=False)
        df.set_index(["ID", "timestamp", "dim"], drop=False, inplace=True)

        # drop ID index
        df = df.droplevel("ID")
        # fix and drop timestamps
        df = df.loc[timestamps]
        df = df.droplevel("timestamp")
        ## fix dims
        df = df.loc[dims]

        # set index back
        df.set_index(["ID", "timestamp", "dim"], drop=True, inplace=True)

        return df[features]

    def get_split_pattern(
        self,
    ):
        return self.metadata.at[self.datatype, "split_pattern"]

    def set_subsplit_pattern(
        self,
        subsplit_pattern: np.ndarray = np.array([]),
        subsplit_pattern_index: np.ndarray = np.array([]),
    ) -> None:
        """Set the subsplit_pattern for the loader to act on.

        If no subsplit_pattern or subsplit_pattern_index is given, load the whole
        split_pattern as the subsplit_pattern.

        Args:
            subsplit_pattern (np.ndarray, optional): The subsplit names to set.
            subsplit_pattern_index (np.ndarray, optional): The subsplit indices to set.

        Raises:
            ValueError: If both `subsplit_pattern` and `subsplit_pattern_index`
                are given as parameters or if they are, respectively,
                invalid for the data's split_pattern.

        """
        if np.size(subsplit_pattern) != 0 and np.size(subsplit_pattern_index) != 0:
            raise ValueError(
                "Give either subsplit_pattern or subsplit_pattern_index, not both."
            )
        elif np.size(subsplit_pattern) == 0 and np.size(subsplit_pattern_index) == 0:
            self.subsplit_pattern = self.metadata.at[self.datatype, "split_pattern"]
        elif np.size(subsplit_pattern_index) != 0:
            split_pattern = self.metadata.at[self.datatype, "split_pattern"]
            if max(subsplit_pattern_index) < len(split_pattern):
                self.subsplit_pattern = [
                    split_pattern[i] for i in subsplit_pattern_index
                ]
            else:
                raise ValueError("Invalid split indices.")
        else:  # subsplit_pattern is not None:
            split_pattern = self.metadata.at[self.datatype, "split_pattern"]
            if set(subsplit_pattern).issubset(set(split_pattern)):
                self.subsplit_pattern = subsplit_pattern
            else:
                raise ValueError("Invalid split names.")

        if len(self.subsplit_pattern) > 0:
            self.current_split = self.subsplit_pattern[0]
        else:
            self.current_split = ""

    def reset_current_split(self, autoload=True) -> None:
        """Reset split index to 0."""
        self.current_split = self.subsplit_pattern[0]
        if autoload:
            self.load()

    def next_current_split(self, autoload=True) -> None:
        """Increment split index by 1."""
        if type(self.subsplit_pattern) is np.ndarray:
            split_index = (
                np.where(self.subsplit_pattern == self.current_split)[0][0] + 1
            )
        elif type(self.subsplit_pattern) is list:
            split_index = self.subsplit_pattern.index(self.current_split) + 1

        if split_index < len(self.subsplit_pattern):
            self.current_split = self.subsplit_pattern[split_index]
            if autoload:
                self.load()
        else:
            raise IndexError("next split out of range")

    def set_current_split(self, new_split: str, autoload=True) -> None:
        """Set the split.

        Args:
            index (int): Value to set the current split index.

        """
        self.current_split = new_split
        if autoload:
            self.load()

    def index_set_current_split(self, index: int, autoload=True) -> None:
        """Set the split index.

        Args:
            index (int): Value to set the current split index.

        """
        self.current_split = self.subsplit_pattern[index]
        if autoload:
            self.load()

    def get_filename(self, for_metadata: bool = False) -> None:
        if for_metadata:
            filename = "metadata"
            if self.parallel:
                filename += "-" + self.current_split
            filename += ".pqt"
            return self._append_path(filename)

        filename = self.datatype + "-" + self.current_split + ".pqt"
        return self._append_path(filename)

    def load(self) -> pd.DataFrame:
        """Load datatatype's data.

        Returns:
            pd.DataFrame: The pandas' data.

        """
        filename = self.get_filename()
        if self.datatype is None or not os.path.isfile(filename):
            self.df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], [], []], names=("ID", "timestamp", "dim")
                )
            )
        else:
            self.df = pd.read_parquet(filename)
        return self.df

    def write(self, write_metadata=True) -> None:
        """Write datatatype's data.

        Raises:
            ValueError: If permission is only 'read' or if attribute `datatype` is not
                defined.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        elif self.datatype is None:
            raise ValueError("No defined datatype.")

        self.df.to_parquet(self.get_filename())

        if write_metadata:
            self.write_metadata()

    def set_df(
        self,
        df: pd.DataFrame = None,
        ID: str = None,
        dim_label: np.ndarray = None,
        timestamp: np.ndarray = None,
        format_df: bool = True,
        update_metadata: bool = True,
    ) -> None:
        """Set datatatype's DataFrame.

        Args:
            df (pd.DataFrame): A dataframe with data for the datatype.

        Raises:
            ValueError: If trying to overwrite data without 'overwrite' permission
                or `df` is not well-defined.

        """
        if len(df) > 0 and self.permission != "overwrite":
            raise ValueError(
                "To change a non-empty datatype, you need 'overwrite' permission."
            )
        if format_df:
            df = convert_to_TSdf(
                data=df,
                ID=ID,
                timestamp=timestamp,
                dim_label=dim_label,
            )
        if update_metadata:
            self.update_metadata_from_df(df)  # upate metadata
        self.df = df

    def rm_datatype(self, rm_from_metadata: bool = True) -> None:
        """Remove datatatype's data.

        Args:
            rm_from_metadata (bool, optional): If the datatype should also be removed
                from metadata. Default is True.

        Raises:
            ValueError: If permission is not overwrite or `self.path` is equal to
                `new_path`.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove a datatype, you need 'overwrite' permission")
        elif self.df.empty:
            raise ValueError("Trying to remove nonexistent datatype.")
        self.df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], [], []], names=("ID", "timestamp", "dim")
            )
        )

        if rm_from_metadata:
            self.metadata.drop(self.datatype, inplace=True)

    #######################
    # add data to dataype #
    #######################

    def get_IDs(self):
        return self.metadata.loc[self.datatype, "IDs"]

    def get_dim_label(self, ID):
        return self.get_df(IDs=[ID]).index.get_level_values("dim").unique()

    def add_data(
        self,
        data=None,
        ID: str = None,
        dim_label: np.ndarray = None,
        timestamp: np.ndarray = None,
        feature_label: np.ndarray = None,
        collision: str = "update",
        format_df: bool = True,
        update_metadata: bool = True,
    ) -> None:
        """Add ID to datatype.

        Caution, it Changes df. `df`'s columns could include "ID", "timestamp",
        "dim". If they don't have either, one will be provided for them.

        Quicker if ID is provided


        If no dim_label is given, assumes the number of dependent dimension is 1.

        Args:
            df (pd.DataFrame): A dataframe with data for a given `ID`.
            ID (str): The unique identication name for the data.
            collision (str, optional): To choose between {'ignore', 'append',
                'update', 'overwerite'}

                - 'update' (default): Updates the value.
                - 'overwrite' : Overwrite the value.
                - 'ignore' : Does nothing.
                - 'append' : Append without index verification df
                   Dangerous: could lead to multiple timestamp problem.

        Raises:
            ValueError: If `ID` is not well-defined or if trying to
                overwrite data without the permission.

        """
        # format data
        if data is None:
            if not self.df.empty and collision == "overwrite":
                self.df = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays(
                        [[], [], []], names=("ID", "timestamp", "dim")
                    )
                )
            return

        if format_df:
            df = convert_to_TSdf(
                data,
                ID=ID,
                timestamp=timestamp,
                dim_label=dim_label,
                feature_label=feature_label,
            )
        else:
            df = data
        if ID is None:
            IDs = np.unique(np.array(df.index.get_level_values("ID")))

        if update_metadata:
            self.update_metadata_from_df(df)  # upate metadata

        if collision == "update" and self.permission == "overwrite":
            self.df = df.combine_first(self.df)
            return

        if ID in self.df.index:
            if collision == "ignore":
                return
            elif collision == "overwrite" and self.permission == "overwrite":
                self.rm_ID(ID, rm_from_metadata=False)  # Keep metadata
                self.df = df.combine_first(self.df)
            else:
                raise ValueError(
                    "Trying to 'overwrite' an ID without permission; "
                    " Or collision parameter not valid"
                )
        else:
            # Append the ID to `self.df`.
            self.df = df.combine_first(self.df)
            # faster?
            # self.df = pd.concat([self.df, df], axis=0)

    def add_feature(
        self, df: pd.DataFrame = None, ID: str = None, feature: str = None
    ) -> None:
        """Add feature to ID in datatype, merging on 'timestamp'.

        This method needs the overwrite permission because you overwrite an ID by
        providing a feature, hence changing the length of the ID.

        If ID is not specify, add it to all datatype. If feature already present and
        not overwrite, gives a warning.  To use `add_feature`, you need overwrite
        permission, because you overwrite the previous features to have the same
        lenght as the added `feature`.

        Args:
            df (pd.DataFrame): A dataframe with a `feature` column for a given `ID`.
            ID (str): The unique identication name for the data.
            feature (str): The feature name for the column.

        Raises:
            ValueError: If `ID`, `feature` or `df` are not well-defined or if trying to
                overwrite data without the permission.

        """
        if df is None or "timestamp" not in df.columns or feature not in df.columns:
            raise ValueError("Need a well-defined DataFrame.")
        elif ID is None:
            raise ValueError("Need an ID.")
        elif feature is None:
            raise ValueError("Need a feature.")

        if self.permission != "overwrite":
            raise ValueError("Trying to `add_feature` without 'overwrite' permission")

        if ID not in self.df.index:
            # ID not in self.df, use the `add_data` method
            self.add_data(df, ID)  # Metadata handled there
        else:
            # ID in self.df, overwrite ID row
            current_ID = self.df.loc[ID].reset_index(drop=False)
            df = df.reset_index(drop=False)
            if "index" in np.array(df.columns):  # remove index column is it's there
                df = df.drop(columns=["index"])
            df_ID = df.combine_first(current_ID)
            # You need to overwrite the ID, to have same input length
            self.add_data(df_ID, ID, collision="overwrite")  # Metadata handled there

    ############################
    # remove data from dataype #
    ############################

    def rm_ID(self, ID: str = None, rm_from_metadata: bool = True) -> None:
        """Remove ID to datatype.

        Args:
            ID (str): The unique identication name for the data.
            rm_from_metadata (bool, optional): If the `ID` should also be removed from
                metadata. Default is True.

        Raises:
            ValueError: If permission is not "overwrite" or if `ID` is not in the
                index of `self.df`.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove an ID, you need 'overwrite' permission.")

        elif ID not in self.df.index:
            raise ValueError("ID does not exsit and trying to remove it.")

        # update df
        self.df.drop(index=ID, level="ID", inplace=True)
        if rm_from_metadata:
            self.set_metadata(IDs=np.array(self.df.index.droplevel(1).unique()))

    def rm_feature(self, feature: str = None, rm_from_metadata: bool = True) -> None:
        """Remove feature to datatype.

        Args:
            feature (str): The feature name for the column.
            rm_from_metadata (bool, optional): If the `feature` should also be removed
                from metadata. Default is True.

        Raises:
            ValueError: If permission is not "overwrite" or if `feature` is not in the
                columns of `self.df`.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove a feature, you need 'overwrite' permission")
        elif feature not in self.df.columns:
            raise ValueError("Trying to remove nonexistent feature.")

        # update df
        self.df.drop(columns=feature, inplace=True)
        if rm_from_metadata:
            self.set_metadata(features=np.array(self.df.columns.unique()))


class LoaderTSdfCSV(LoaderTSdf):
    def get_filename(self, for_metadata: bool = False) -> None:
        if for_metadata:
            filename = "metadata"
            if self.parallel:
                filename += "-" + self.datatype + "-" + self.current_split
            filename += ".csv"
            return self._append_path(filename)

        filename = self.datatype + "-" + self.current_split + ".csv"
        return self._append_path(filename)

    def load_metadata(self) -> pd.DataFrame:
        """Load datataset's metadata.

        Returns:
            pd.DataFrame: The pandas' metadata.

        """
        metadata_file = self.get_filename(for_metadata=True)
        if os.path.isfile(metadata_file):
            self.metadata = pd.read_csv(metadata_file)
        else:
            self.metadata = pd.DataFrame()

        return self.metadata

    def write_metadata(self) -> None:
        """Write datataset's metadata.

        Raises:
            ValueError: If permission is read.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        self.metadata.to_csv(self.get_filename(for_metadata=True))

    def load(self) -> pd.DataFrame:
        """Load datatatype's data.

        Returns:
            pd.DataFrame: The pandas' data.

        """
        filename = self.get_filename()
        if self.datatype is None or not os.path.isfile(filename):
            self.df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], [], []], names=("ID", "timestamp", "dim")
                )
            )
        else:
            self.df = pd.read_csv(filename)
        return self.df

    def write(self, write_metadata=True) -> None:
        """Write datatatype's data.

        Raises:
            ValueError: If permission is only 'read' or if attribute `datatype` is not
                defined.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        elif self.datatype is None:
            raise ValueError("No defined datatype.")

        self.df.to_csv(self.get_filename())

        if write_metadata:
            self.write_metadata()

    def merge_metadata(self, write_metadata: bool = True, rm: bool = True) -> None:
        """Merge metadata between 'metadata-' file.

        Args:
            write (bool, optional): Whether or not to write the merged metadata.
            rm (bool, optional): Whether or not to removed all the 'metadata-' files
                after the merge.

        Raises:
            ValueError: If trying to write metadata with 'read'
                permission; Or if trying to remove metadata (on disk or
                memory) without 'overwrite' permission; Or if `parallel`
                attribute is `True`.

        """
        if self.permission == "read" and self.write:
            raise ValueError(
                "You cannot write metadata while merging " + "with 'read' permission."
            )
        elif self.permission != "overwrite" and rm:
            raise ValueError(
                "You cannot remove metadata while merging "
                + "without 'overwrite' permission."
            )

        elif not self.metadata.empty and self.permission != "overwrite":
            raise ValueError(
                "Trying to merge metadata but it already exists. "
                + "To force it, change permission to 'overwrite'."
            )

        if self.parallel:
            raise ValueError(
                "Set the parallel execution attribute " + "to `False` before merging."
            )

        datatype_keep = self.datatype
        self.metadata = pd.DataFrame()
        for filename in os.listdir(self.path):
            if filename[0:9] == "metadata-":
                metadata_file = self._append_path(filename)
                new_metadata = pd.read_csv(metadata_file)
                for datatype in new_metadata.index:
                    self.datatype = datatype
                    self._add_datatype_to_metadata()
                    features = new_metadata.loc[self.datatype, "features"]
                    IDs = new_metadata.loc[self.datatype, "IDs"]
                    split_pattern = new_metadata.loc[self.datatype, "split_pattern"]

                    self.append_to_metadata(
                        split_pattern=split_pattern, IDs=IDs, features=features
                    )

                # remove metadata-* file
                if rm:
                    os.remove(metadata_file)

        self.datatype = datatype_keep
        if write_metadata and not self.metadata.empty:
            self.write_metadata()


class LoadersProcess:
    """A collection of loaders and a function to apply to them using multiprocessing.


    Need to respect:
    - n_procs + n_loaders <= n threads
    - 2 * n_loaders <= n threads

    Args:
        loaders (TSLoader): Sets attribute of the same name.
        function Callable[[TSloader], None]): Sets attribute of the same name.

    Attributes:
        loaders ("TSLoader"): list of loaders to use with their split_pattern for
            multiprocessing.
        df_function Callable[["TSloader"], None]): A function to apply to every a df
            every loaders.
        loader_function Callable[["TSloader"], None]): A function to apply to every split_pattern of
            every loaders.

    """

    def __init__(
        self,
        data_path=None,
        output_path=None,
        datatype="",
        output_datatype=None,
        n_procs=1,
        n_loaders=1,
        IDs=None,
        subsplit_pattern=None,
        loader_function: Callable[["TSloader"], None] = None,
        df_function: Callable[[pd.DataFrame], None] = None,
        loaders: "TSloader" = None,
        autoload=True,
        parallel=True,
    ):
        """Init method."""

        def df_function_wrap(loader, IDs):
            if type(IDs) is not list:
                IDs = [IDs]
            try:
                df = loader.get_df(IDs=IDs)
            except KeyError:
                print(IDs, "not in ", loader.current_split, "data")
                return

            split = loader.current_split + "_" + "".join(IDs)
            output_loader = LoaderTSdf(
                path=self.output_path,
                datatype=self.output_datatype,
                split_pattern=split,
                parallel=self.parallel,
            )
            df_function(df, output_loader)

        self.set_loaders(
            data_path=data_path,
            datatype=datatype,
            subsplit_pattern=subsplit_pattern,
            n_loaders=n_loaders,
            loaders=loaders,
        )

        self.datatype = self.loaders[0].datatype
        self.data_path = self.loaders[0].path

        if output_path is None:
            output_path = self.data_path
        if output_datatype is None:
            output_datatype = self.datatype

        if loader_function is None:
            loader_function = lambda loader: None
        if df_function is None:
            df_function = lambda split, ID, df: None

        if IDs is None:
            IDs = self.loaders[0].get_IDs()

        self.loader_function = loader_function
        self.df_function = df_function_wrap
        self.n_procs = n_procs
        self.output_path = output_path
        self.output_datatype = output_datatype
        self.IDs = IDs
        self.autoload = autoload
        self.parallel = parallel

    def run_ID(self, merge_data=False):
        """For every ID, apply `df_function` attribute optionally in parallel."""

        def df_ID_function(loader):
            for split in loader.subsplit_pattern:
                loader.set_current_split(split, autoload=self.autoload)
                # run df_function in parallel with the available n_procs
                if self.parallel:
                    Parallel(n_jobs=self.n_procs)(
                        delayed(self.df_function)(loader, IDs) for IDs in self.IDs
                    )
                else:
                    for IDs in self.IDs:
                        self.df_function(loader, IDs)

        if self.parallel:
            Parallel(n_jobs=len(self.loaders))(
                delayed(df_ID_function)(loader) for loader in self.loaders
            )
        else:
            for loader in self.loaders:
                df_ID_function(loader)

        if self.parallel and merge_data:
            output_loader = LoaderTSdf(
                path=self.output_path, datatype=self.output_datatype
            )
            output_loader.merge_metadata()
            LoaderTSdf.merge_splitted_files(output_loader, len(self.loaders))

    def run_loader(self):
        def loaders_split_function(loader):
            for split in loader.subsplit_pattern:
                loader.set_current_split(split, autoload=self.autoload)
                self.loader_function(loader)

        if self.parallel:
            Parallel(n_jobs=len(self.loaders))(
                delayed(loaders_split_function)(loader) for loader in self.loaders
            )
        else:
            for loader in self.loaders:
                loaders_split_function(loader)

    def set_loaders(
        self,
        data_path=None,
        datatype="",
        subsplit_pattern=None,
        n_loaders=1,
        loaders: "TSloader" = None,
    ):
        if loaders is None:
            if data_path is None:
                raise ValueError("Need data_path")
            metaloader = LoaderTSdf(path=data_path, datatype=datatype)

            if subsplit_pattern is None:
                subsplit_pattern = metaloader.get_split_pattern()
            last_split_index = len(subsplit_pattern)

            if last_split_index == 0:
                raise ValueError("split_pattern not defined.")
            if n_loaders > last_split_index:
                raise ValueError("n_loaders greater than length of subsplit pattern.")

            loaders = []
            for i in range(n_loaders - 1):
                subsplit_index = slice(
                    *[last_split_index // n_loaders * j for j in [i, i + 1]]
                )
                loader = LoaderTSdf(
                    path=data_path,
                    datatype=datatype,
                    subsplit_pattern=subsplit_pattern[subsplit_index],
                    autoload=False,
                )
                loaders.append(loader)
            # i == n_loaders
            subsplit_index = slice(
                last_split_index // n_loaders * (n_loaders - 1), last_split_index
            )
            loader = LoaderTSdf(
                path=data_path,
                datatype=datatype,
                subsplit_pattern=subsplit_pattern[subsplit_index],
                autoload=False,
            )
            loaders.append(loader)

        self.loaders = loaders
