from __future__ import annotations

import ast
import logging
import math
import os
import re
import shutil
from typing import Callable, Optional, Type

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from TSbench.TSdata.data import AnyData, Data
from TSbench.TSdata.DataFormat import convert_to_TSdf, convert_from_TSdf


class TSloader:
    path: str
    df: pd.DataFrame
    datatype: str
    subsplit_pattern: np.ndarray
    current_split: str
    permission: str
    autoload: bool

    def __init__(
        self,
        path: str = "data/",
        datatype: Optional[str] = None,
        split_pattern: Optional[np.ndarray] = None,
        subsplit_pattern: Optional[np.ndarray] = None,
        subsplit_pattern_index: Optional[np.ndarray] = None,
        permission: str = "overwrite",
        autoload: bool = True,
    ) -> None:
        """Init method."""
        if split_pattern is None:
            split_pattern = np.array([])
        if subsplit_pattern is None:
            subsplit_pattern = np.array([])
        if subsplit_pattern_index is None:
            subsplit_pattern_index = np.array([])
        if datatype is None:
            raise ValueError("Give a datatype.")

        # Initialize basic parameters
        self.set_permission(permission)  # read, write, overwrite
        self.autoload = autoload

        # Set path
        self.set_path(path)

        # Initialize metadata
        self.load_metadata()

        ## Set the datatype
        self.set_datatype(
            datatype, split_pattern, subsplit_pattern, subsplit_pattern_index
        )
        # Initialize df
        if self.autoload:
            self.df = self.load()
        else:
            self.df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], [], []], names=("ID", "timestamp", "dim")
                )
            )

    def set_path(self, path: str, ensure_path=True) -> None:
        """Set the current path.

        Args:
            path (str): The path to set.
        """
        self.path = path
        if ensure_path:
            self._create_path()

    def set_permission(self, permission: str = "write") -> None:
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

    def rm_dataset(self, ignore_errors: bool = True) -> None:
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

    def restart_dataset(self) -> None:
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

        metadata_file = self.get_filename(for_metadata=True)

        old_path = self.path
        # create new path
        self.set_path(new_path)

        new_metadata_file = self.get_filename(for_metadata=True)
        shutil.copyfile(metadata_file, new_metadata_file)

        # autoload is a waste of time here
        autoload_value = self.autoload
        self.autoload = False
        for split in self.get_split_pattern():
            self.set_current_split(split)
            dst = self.get_filename()
            src = os.path.join(old_path, os.path.basename(dst))
            shutil.copyfile(src, dst)
        # set autoload to original value
        self.autoload = autoload_value

    def _create_path(self) -> None:
        """Create the dataset if it doesn't exsist."""
        # if path doesn't exsist.
        if self.path != "" and not os.path.isdir(self.path):
            # Create path
            if self.permission != "read":
                logging.info(f"Path '{self.path}' does not exist, creating.")
                # exist_ok is True because of possible parllel call to makedirs.
                os.makedirs(self.path, exist_ok=True)
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

    def list_datatypes(self) -> np.ndarray:
        """List datatypes from dataset."""
        return np.unique(np.array(self.metadata.index))

    def _add_datatype_to_metadata(self) -> None:
        """Add the current datatype to the metadata indices."""
        if self.metadata.empty:
            self.metadata = pd.DataFrame(
                columns=pd.Index(["datatype", "IDs", "features", "split_pattern"])
            )
            self.metadata["datatype"] = [self.datatype]
            self.metadata.set_index(["datatype"], inplace=True, drop=True)

            self.metadata.at[self.datatype, "IDs"] = np.array([])
            self.metadata.at[self.datatype, "features"] = np.array([])
            self.metadata.at[self.datatype, "split_pattern"] = np.array([])
        elif self.datatype not in self.metadata.index:
            # datatype = self.metadata.index.append(pd.Index([self.datatype]))
            # self.metadata = self.metadata.reindex(datatype, fill_value=np.array([]))
            new_indices = self.metadata.index.append(pd.Index([self.datatype]))
            self.metadata = self.metadata.reindex(new_indices)
            self.metadata.loc[self.datatype] = [
                np.array([]) for _ in range(self.metadata.shape[1])
            ]
        # else datatype is already in metadata indices

    def _update_split_pattern_to_metadata(
        self, split_pattern: Optional[np.ndarray] = None
    ) -> None:
        """Set split pattern in "metadata"

        Update split_pattern in metadata if needed, otherwise do nothing.

        Args:
            split_pattern (np.ndarray, optional):

        Raises:
            ValueError: If split_pattern exists and no overwrite permission is granted.

        """
        if split_pattern is None:
            split_pattern = np.array([])

        # if split_pattern is empty
        if np.size(split_pattern) == 0:
            if np.size(self.metadata.at[self.datatype, "split_pattern"]) != 0:
                # split pattern already defined in metadata, do nothing
                return
            split_pattern = np.array([])
        self.set_metadata(split_pattern=split_pattern)

    def set_metadata(self, **metadata: list | np.ndarray) -> None:
        """set metadata.

        Args:
            **metadata (np.ndarray):

        Raises:
            ValueError: If permission is not overwrite.

        """
        for key in metadata:
            entry = np.array(metadata[key], ndmin=1)
            # if entry.dtype == "<U6":

            if key not in self.metadata.columns:
                self.metadata[key] = pd.Series(data=np.array([], dtype=entry.dtype))
            elif np.size(self.metadata[key]) > 0 and self.permission != "overwrite":
                raise ValueError(
                    f"{key} already exsists in metadata. "
                    + "To overwite it, you need the overwrite permission."
                )
            self.metadata.at[self.datatype, key] = entry

            # if ndim and length is 1, pandas reduce it to dim to 0.
            # This brings it back as it is suppose to be: ndim == 1.
            if entry.ndim == 1 and len(entry) == 1:
                self.metadata = self.metadata.map(lambda arr: np.array(arr, ndmin=1))

    def update_metadata(self) -> None:
        """Update metadata using df."""
        self.append_to_metadata(
            IDs=np.unique(np.array(self.df.index.get_level_values("ID")))
        )
        self.append_to_metadata(features=np.array(self.df.columns))

    def update_split_pattern_from_dataset(self) -> None:
        """Update metadata using df."""
        split_pattern = []
        for filename in os.listdir(self.path):
            if filename[0 : len(self.datatype)] == self.datatype:
                filename = os.path.splitext(filename)[0]
                split_pattern.append(filename[len(self.datatype) + 1 :])

        split_pattern = np.sort(split_pattern)
        self.set_metadata(split_pattern=split_pattern)

    def update_metadata_from_dataset(self) -> None:
        """Update metadata using dataset."""
        for datatype in self.list_datatypes():
            self.set_datatype(datatype)

            self.update_split_pattern_from_dataset()
            self.df = self.load()

            self.update_metadata()

        if not self.metadata.empty:
            self.write_metadata()

    def append_to_metadata(self, **metadata: list | np.ndarray) -> None:
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

    def merge_splitted_metadata(
        self, write_metadata: bool = True, rm: bool = True
    ) -> None:
        """Merge metadata between 'metadata-' file.

        Args:
            write (bool, optional): Whether or not to write the merged metadata.
            rm (bool, optional): Whether or not to removed all the 'metadata-' files
                after the merge.

        Raises:
            ValueError: If trying to write metadata with 'read'
                permission; Or if trying to remove metadata (on disk or
                memory) without 'overwrite' permission.

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

        initial_datatype = self.datatype
        self.metadata = pd.DataFrame()  # needed for DatasetOperations.merge_dataset
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

        self.datatype = initial_datatype

        if write_metadata and not self.metadata.empty:
            self.write_metadata()

    @staticmethod
    def merge_splitted_data(
        loader, n_jobs, write_metadata: bool = True, rm: bool = True
    ) -> None:
        if loader.permission == "read":
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

        def merge_day_wise(loader: "TSloader", new_split: str) -> None:
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

        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(
                delayed(merge_day_wise)(loader, new_split)
                for new_split in new_split_pattern
            )

        if write_metadata:
            metadata_loader = LoaderTSdf(path=loader.path, datatype=loader.datatype)
            metadata_loader.set_metadata(split_pattern=new_split_pattern)
            metadata_loader.write_metadata()

    def load_metadata(self) -> pd.DataFrame:
        """Load datataset's metadata.

        Returns:
            pd.DataFrame: The pandas' metadata.

        """
        metadata_file = self.get_filename(for_metadata=True)
        if os.path.isfile(metadata_file):
            self.metadata = pd.read_parquet(metadata_file)
        else:
            self.metadata = pd.DataFrame(index=pd.Index(["datatype"]))
        return self.metadata

    def write_metadata(self) -> None:
        """Write datataset's metadata.

        Raises:
            ValueError: If permission is read.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        self.metadata.to_parquet(self.get_filename(for_metadata=True))

    def set_datatype(
        self,
        datatype: str,
        split_pattern: Optional[np.ndarray] = None,
        subsplit_pattern: Optional[np.ndarray] = None,
        subsplit_pattern_index: Optional[np.ndarray] = None,
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
        if split_pattern is None:
            split_pattern = np.array([])
        if subsplit_pattern is None:
            subsplit_pattern = np.array([])
        if subsplit_pattern_index is None:
            subsplit_pattern_index = np.array([])

        self.datatype = datatype

        # update metadata
        self._add_datatype_to_metadata()
        self._update_split_pattern_to_metadata(split_pattern)

        # set subsplit_pattern for loader
        self.set_subsplit_pattern(subsplit_pattern, subsplit_pattern_index)

    def train_test_split(
        self,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
        rounding: Optional[str] = "before",
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame]:
        if train_size is None:
            if test_size is None:
                test_size = 0.3
                train_size = 0.7
            else:
                train_size = 1 - test_size
        if test_size is None:
            test_size = 1 - train_size
        if test_size is None:
            test_size = 1 - train_size

        if train_size + test_size != 1:
            raise ValueError("The sum of 'train_size' and 'test_size' must be 1.")

        split_index = train_size * self.get_df().shape[0]
        if rounding == "before":
            split_index = math.floor(split_index)
        elif rounding == "after":
            split_index = math.ceil(split_index)
        else:
            raise ValueError("rounding is either 'before' or 'after'.")

        train = self.get_df(end_index=split_index)
        test = self.get_df(start_index=split_index + 1)
        return train, test

    def get_df(
        self,
        start: Optional[int | str] = None,
        start_index: Optional[int] = None,
        end: Optional[int | str] = None,
        end_index: Optional[int] = None,
        IDs: Optional[np.ndarray | str] = None,
        timestamps: Optional[np.ndarray] = None,
        dims: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Alias for get_timeseries

        Use get_timeseries instead. About to be deprecated.
        """
        if isinstance(IDs, str):
            IDs = np.array([IDs])
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
        self,
        start: Optional[int | str] = None,
        start_index: Optional[int] = None,
        end: Optional[int | str] = None,
        end_index: Optional[int] = None,
        IDs: Optional[slice | np.ndarray] = None,
        unique: bool = False,
    ) -> np.ndarray:
        """Get all timestamps for the given IDs.

        If multiple IDs are given (or all by default), timestamps are repeated
        """
        if start is not None and start_index is not None:
            raise ValueError("Either give start or a start_index")
        elif end is not None and end_index is not None:
            raise ValueError("Either give end or a end_index")

        if IDs is None:
            IDs = slice(None)

        # get all timestamps
        dim_dropped_df = self.df.loc[IDs, :, self.get_dim_label()[0]]
        timestamps = np.array(dim_dropped_df.index.get_level_values("timestamp"))

        if unique:
            timestamps = np.unique(timestamps)

        # If len(IDs) == 1, timestamps is sorted; else,
        if isinstance(IDs, slice) or len(IDs) != 1:
            timestamps = np.sort(timestamps)

        if start is not None:
            start_index = int(np.searchsorted(timestamps, start))

        if end is not None:
            end_index = int(np.searchsorted(timestamps, end))

        return timestamps[start_index:end_index]

    def get_timeseries(
        self,
        start: Optional[int | str] = None,
        start_index: Optional[int] = None,
        end: Optional[int | str] = None,
        end_index: Optional[int] = None,
        IDs: Optional[slice | np.ndarray] = None,
        timestamps: Optional[slice | np.ndarray] = None,
        dims: Optional[slice | np.ndarray] = None,
        features: Optional[slice | np.ndarray] = None,
        tstype: Type[Data] = pd.DataFrame,
    ) -> Data:
        """Get DataFrame for the datatype.

        Much more efficient if IDs is provided and list is short

        If "IDs", "timestamps" or "dims" is specify, fix that value. Otherwise
        returns the entries corrresponding to all values. The more efficient in
        memory is to have, at most, one specific fix value.
        """
        args = [start, start_index, end, end_index, IDs, timestamps, dims, features]
        if self.df.empty or all(arg is None for arg in args):
            return convert_from_TSdf(self.df, tstype)

        if timestamps is not None and any(arg is not None for arg in args[0:4]):
            raise ValueError("Give either timestamp or start/end, not both.")

        # timestamps related args are not all None
        if any(arg is not None for arg in args[0:4]):
            timestamps = self.get_timestamp(
                start=start,
                start_index=start_index,
                end=end,
                end_index=end_index,
                IDs=IDs,
            )
        elif timestamps is None:
            timestamps = slice(None)

        if IDs is None:
            IDs = slice(None)
        if dims is None:
            dims = slice(None)
        if features is None:
            features = slice(None)

        return convert_from_TSdf(self.df.loc[IDs, timestamps, dims][features], tstype)
        # What follows does
        # self.df.loc[IDs, timestamps, dims][features]
        # much more quickly but create a lot of DataFrames (using copy)
        # if IDs is None:
        #     IDs = slice(None)
        # if dims is None:
        #     dims = slice(None)
        # if features is None:
        #     features = slice(None)

        # # keep index information in columns
        # df = self.df
        # df = df.reset_index(drop=False)
        # df.set_index(["ID", "timestamp", "dim"], drop=False, inplace=True)

        # # fix and drop ID index
        # df = df.loc[IDs]
        # df = df.droplevel("ID")
        # # fix and drop timestamps
        # df = df.loc[timestamps]
        # df = df.droplevel("timestamp")
        # # fix dims, no need to drop
        # df = df.loc[dims]

        # # set index back
        # df.set_index(["ID", "timestamp", "dim"], drop=True, inplace=True)

        # return convert_from_TSdf(df[features], tstype)

    def concat_subsplit_pattern(self):
        df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], [], []], names=("ID", "timestamp", "dim")
            )
        )
        for split in self.get_split_pattern():
            self.set_current_split(split)
            df = pd.concat([df, self.df], axis=0)
        self.df = df

    def get_split_pattern(self) -> np.ndarray:
        return self.metadata.at[self.datatype, "split_pattern"]

    def set_subsplit_pattern(
        self,
        subsplit_pattern: Optional[np.ndarray] = None,
        subsplit_pattern_index: Optional[np.ndarray] = None,
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
        if subsplit_pattern is None:
            subsplit_pattern = np.array([])
        if subsplit_pattern_index is None:
            subsplit_pattern_index = np.array([])

        if np.size(subsplit_pattern) != 0 and np.size(subsplit_pattern_index) != 0:
            raise ValueError(
                "Give either subsplit_pattern or subsplit_pattern_index, not both."
            )
        elif np.size(subsplit_pattern) == 0 and np.size(subsplit_pattern_index) == 0:
            self.subsplit_pattern = self.metadata.at[self.datatype, "split_pattern"]
        elif np.size(subsplit_pattern_index) != 0:
            split_pattern = self.metadata.at[self.datatype, "split_pattern"]
            if max(subsplit_pattern_index) < len(split_pattern):
                self.subsplit_pattern = np.array(
                    [split_pattern[i] for i in subsplit_pattern_index]
                )
            else:
                raise ValueError("Invalid split indices.")
        else:  # subsplit_pattern is not None:
            split_pattern = self.metadata.at[self.datatype, "split_pattern"]
            if set(subsplit_pattern).issubset(set(split_pattern)):
                self.subsplit_pattern = subsplit_pattern
            else:
                raise ValueError(
                    "Invalid sub split names : "
                    f"{set(subsplit_pattern) - set(split_pattern)}"
                )

        if len(self.subsplit_pattern) > 0:
            self.current_split = self.subsplit_pattern[0]
        else:
            self.current_split = ""

    def reset_current_split(self) -> None:
        """Reset split index to 0."""
        self.current_split = self.subsplit_pattern[0]
        if self.autoload:
            self.load()

    def next_current_split(self) -> None:
        """Increment split index by 1."""
        if isinstance(self.subsplit_pattern, np.ndarray):
            split_index: int = (
                np.where(self.subsplit_pattern == self.current_split)[0][0] + 1
            )
        elif isinstance(self.subsplit_pattern, list):
            split_index = self.subsplit_pattern.index(self.current_split) + 1

        if split_index < len(self.subsplit_pattern):
            self.current_split = self.subsplit_pattern[split_index]
            # self.current_split = self.subsplit_pattern[0]
            if self.autoload:
                self.load()
        else:
            raise IndexError("next split out of range")

    def set_current_split(self, new_split: str) -> None:
        """Set the split.

        Args:
            index (int): Value to set the current split index.

        """
        self.current_split = new_split
        if self.autoload:
            self.load()

    def index_set_current_split(self, index: int) -> None:
        """Set the split using index.

        Args:
            index (int): Value to set the current split index.

        """
        self.current_split = self.subsplit_pattern[index]
        if self.autoload:
            self.load()

    def get_filename(self, for_metadata: bool = False) -> str:
        if for_metadata:
            filename = "metadata.pqt"
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

    def write(self, write_metadata: bool = True) -> None:
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
        df: Optional[pd.DataFrame] = None,
        ID: Optional[str] = None,
        dim_label: Optional[np.ndarray] = None,
        timestamp: Optional[np.ndarray] = None,
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
        if df is None:
            df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], [], []], names=("ID", "timestamp", "dim")
                )
            )
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

        self.df = df

        if update_metadata:
            # upate metadata
            self.update_metadata()

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

    def get_IDs(self, from_metadata=False) -> np.ndarray:
        if from_metadata:
            return self.metadata.loc[self.datatype, "IDs"]
        else:
            return np.unique(self.get_df().index.get_level_values("ID"))

    def get_dim_label(self, ID: Optional[str] = None) -> pd.Index:
        if ID is None:
            return self.get_df().index.get_level_values("dim").unique()
        else:
            return (
                self.get_df(IDs=np.array([ID])).index.get_level_values("dim").unique()
            )

    def add_data(
        self,
        data: Optional[AnyData] = None,
        ID: Optional[str] = None,
        dim_label: Optional[np.ndarray] = None,
        timestamp: Optional[np.ndarray] = None,
        feature_label: Optional[np.ndarray] = None,
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
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("data must either be a DataFrame or be formated.")

        if collision == "update" and self.permission == "overwrite":
            self.df = df.combine_first(self.df)
            self.update_metadata()  # upate metadata
            return

        if ID in self.df.index:
            if collision == "ignore":
                return
            elif collision == "overwrite" and self.permission == "overwrite":
                if len(self.get_IDs()) == 1:
                    self.rm_ID(ID, rm_from_metadata=False)  # Keep metadata
                    self.df = df
                else:
                    self.rm_ID(ID, rm_from_metadata=False)  # Keep metadata
                    self.df = df.combine_first(self.df)
            else:
                raise ValueError(
                    "Trying to 'overwrite' an ID without permission; "
                    " Or collision parameter not valid"
                )
        else:
            # Append the ID to `self.df`.
            # self.df = df.combine_first(self.df)
            # faster?
            self.df = pd.concat([self.df, df], axis=0)

        if update_metadata:
            self.update_metadata()  # upate metadata

    def add_feature(
        self,
        df: Optional[pd.DataFrame] = None,
        ID: Optional[str] = None,
        feature: Optional[str] = None,
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

    def rm_ID(self, ID: Optional[str] = None, rm_from_metadata: bool = True) -> None:
        """Remove data conresponding to ID from dataype.

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
            raise ValueError("ID does not exsist and trying to remove it.")

        # update df
        self.df.drop(index=ID, level="ID", inplace=True)
        if rm_from_metadata:
            self.set_metadata(IDs=np.array(self.df.index.droplevel(1).unique()))

    def rm_feature(
        self, feature: Optional[str] = None, rm_from_metadata: bool = True
    ) -> None:
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


class LoaderTSdf(TSloader):
    """Use to write, load and modify a timeseries dataset.

    A TSloader is assigned a path to a "dataset". Optionally, it can
    have a "datatype" which informs about the structure of the data.
    "Datatype" is a collection of multiple input with different "IDs".
    A given "datatype" as the exact same "features" which is the data
    indexed with a "timestamp" (the timeseries).

    A "datatype" can be splitted on different files on disk, this is
    called a "split_pattern". A TSloader with that "datatype" and a
    "subsplit" (either with names or indices) can manipulate the data
    from the files. It is used when a single datatype is too large or
    for parallelization purposes.

    The split_pattern is an "attribute" of a the datatype, stored in
    metadata. It can be updated if new data of split patterns are
    needed, but it is more fixed as a way to name the file for the
    data. It is stored in memory as metadata.

    On the other hand, the subsplit_pattern, is more dynamic, it
    depends on the specific TSloader used to load the data. It tells
    the TSloader the (sub)-files to load for a specific datatype. For
    an example see "example_multiprocess.ipynb".

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
        subsplit_pattern_index (np.narray , optional): The indices to
            use as subsplit_pattern. Default is to use all the indices
            from the split.
        permission (str): To choose between {'read', 'write', 'overwerite'},
            with an increasing level of permission for the loader. The options are :

            Permission is seen as permission for operations on disk.
            - 'read' : Read only the data on disk and change it on memory.
            - 'write' : Add only new datatype and new ID. Any operation that \
               would remove or change data or metadata will raise an error.
            - 'overwrite' (default) :  Do all operations and write on disk.

    """

    def __init__(self, **TSloader_args) -> None:
        """Init method."""
        # Permissions
        super().__init__(**TSloader_args)

    def set_path(self, path: str, ensure_path: bool = True) -> None:
        """Set the current path.

        Args:
            path (str): The path to set.
        """
        self.path = path
        if ensure_path:
            self._create_path()


class LoaderTSdfCSV(LoaderTSdf):
    def get_filename(self, for_metadata: bool = False) -> str:
        if for_metadata:
            filename = "metadata.csv"
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
            self.metadata.set_index(["datatype"], inplace=True, drop=True)
            # pd.read_csv gives string, parse it as np.ndarray
            for index in self.metadata.index:
                for column in self.metadata.columns:
                    arr = self.metadata.at[index, column]
                    if isinstance(arr, str):
                        # parse string
                        arr = re.sub(pattern=" ", repl=",", string=arr)
                        arr = np.array(ast.literal_eval(arr))
                        # change value to np.ndarray
                        self.metadata.at[index, column] = arr
        else:
            self.metadata = pd.DataFrame(
                columns=pd.Index(["IDs", "features", "split_pattern"]),
            )

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
        if self.datatype is None or not os.path.isfile(filename) or not self.autoload:
            self.df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], [], []], names=("ID", "timestamp", "dim")
                )
            )
        else:
            self.df = pd.read_csv(filename)

        return self.df

    def write(self, write_metadata: bool = True) -> None:
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


class LoadersProcess:
    """A collection of loaders and a function to apply to them using multiprocessing.


    Need to respect:
    - n_jobs + n_input_loaders <= n threads

    Args:
        loaders (TSLoader): Sets attribute of the same name.
        function Callable[[TSloader], None]): Sets attribute of the same name.

    Attributes:
        loaders ("TSLoader"): list of loaders to use with their split_pattern for
            multiprocessing.
        df_function Callable[["TSloader"], None]): A function to apply to every a df
            every loaders.
        process_df Callable[["TSloader"], None]): How to process the DataFrame of every
        loader, ID-wise.
        df of every loaders.
        process_split Callable[["TSloader"], None]): How to process the loader for every
        split.


    """

    data_path: str
    output_path: str
    datatype: str
    output_datatype: str
    n_jobs: int
    n_input_loaders: int
    IDs: np.ndarray
    process_split: Optional[Callable[[TSloader], None]]
    process_df: Callable[[pd.DataFrame], pd.DataFrame]
    input_loaders: list["TSloader"]
    autoload: bool

    def __init__(
        self,
        data_path: Optional[str] = None,
        datatype: Optional[str] = None,
        IDs: Optional[np.ndarray] = None,
        subsplit_pattern: Optional[np.ndarray] = None,
        autoload: bool = True,
        input_loaders: Optional[list["TSloader"]] = None,
        output_loader: Optional["TSloader"] = None,
        n_input_loaders: int = 1,
        n_jobs: int = 1,
        process_split: Optional[Callable[["TSloader"], None]] = None,
        process_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        if process_df is None:

            def process_df_default(df: pd.DataFrame) -> pd.DataFrame:
                return df

            process_df = process_df_default

        """Init method."""
        self.n_jobs = n_jobs
        self.process_split = process_split
        self.process_df = process_df

        self.set_input_loaders(
            input_loaders=input_loaders,
            data_path=data_path,
            datatype=datatype,
            subsplit_pattern=subsplit_pattern,
            n_input_loaders=n_input_loaders,
            autoload=autoload,
        )

        self.set_output_loader(
            output_loader=output_loader,
            output_path=self.input_loaders[0].path,
            output_datatype=self.input_loaders[0].datatype,
            subsplit_pattern=subsplit_pattern,
        )

        IDs_from_loader = self.input_loaders[0].get_IDs()
        if IDs is None:
            self.IDs = IDs_from_loader
        else:
            self.IDs = np.intersect1d(IDs, IDs_from_loader)

    def set_input_loaders(
        self,
        input_loaders: Optional[list["TSloader"]] = None,
        data_path: Optional[str] = None,
        datatype: Optional[str] = None,
        subsplit_pattern: Optional[np.ndarray] = None,
        n_input_loaders: int = 1,
        autoload: bool = False,
    ) -> None:
        if input_loaders is None:
            if data_path is None:
                raise ValueError("Need data_path")
            metadata_loader = LoaderTSdf(path=data_path, datatype=datatype)

            if subsplit_pattern is None:
                subsplit_pattern = metadata_loader.get_split_pattern()
            last_split_index = len(subsplit_pattern)

            if last_split_index == 0:
                raise ValueError("split_pattern not defined.")
            if n_input_loaders > last_split_index:
                raise ValueError(
                    "n_input_loaders greater than length of subsplit pattern."
                )

            input_loaders = []
            for i in range(n_input_loaders - 1):
                subsplit_index = slice(
                    *[last_split_index // n_input_loaders * j for j in [i, i + 1]]
                )
                input_loader = LoaderTSdf(
                    path=data_path,
                    datatype=datatype,
                    subsplit_pattern=subsplit_pattern[subsplit_index],
                    autoload=autoload,
                )
                input_loaders.append(input_loader)
            # i == n_input_input_loaders
            subsplit_index = slice(
                last_split_index // n_input_loaders * (n_input_loaders - 1),
                last_split_index,
            )
            input_loader = LoaderTSdf(
                path=data_path,
                datatype=datatype,
                subsplit_pattern=subsplit_pattern[subsplit_index],
                autoload=autoload,
            )
            input_loaders.append(input_loader)
        self.input_loaders = input_loaders

    def set_output_loader(
        self,
        output_loader,
        output_path: str,
        output_datatype,
        subsplit_pattern,
    ):
        if output_loader is None:
            output_loader = LoaderTSdf(
                path=output_path,
                datatype=output_datatype,
                split_pattern=subsplit_pattern,
            )
        self.output_loader = output_loader

    def process_loader_ID(self, input_loader: TSloader, ID: np.ndarray) -> pd.DataFrame:
        """Single fix loader and single fix ID to process.

        Helper method to be called in parallel with a list of IDs and
        a fix input_loader.
        """
        return self.process_df(input_loader.get_df(IDs=ID))

    def process_loader(self, input_loader: "TSloader", write=True) -> None:
        """Single loader to process split-wise and ID-wise.

        Helper method to be called in parallel with a list of input_loaders.
        """
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for split in input_loader.subsplit_pattern:
                input_loader.set_current_split(split)
                self.output_loader.set_current_split(split)
                if self.process_split is not None:
                    self.process_split(input_loader)
                # run df_function in parallel with the available n_jobs
                processed_data_ID = list(
                    parallel(
                        delayed(self.process_loader_ID)(input_loader, IDs)
                        for IDs in self.IDs
                    )
                )
                # merge outputs
                self.output_loader.set_df(
                    pd.concat(processed_data_ID, axis=0),  # type: ignore
                    update_metadata=False,
                )
                if write:
                    self.output_loader.write(write_metadata=False)
            if write:
                self.output_loader.update_metadata_from_dataset()

    def run_process(self, write=True) -> None:
        """For every ID, apply `df_function` attribute in parallel."""
        with Parallel(n_jobs=len(self.input_loaders)) as parallel:
            parallel(
                delayed(self.process_loader)(loader, write=write)
                for loader in self.input_loaders
            )
