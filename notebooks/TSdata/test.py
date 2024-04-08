# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: TSbench
#     language: python
#     name: tsbench
# ---

# %% [markdown]
# # Example TSdata
# This is an example of how to use TSdata.

# %%% [markdown]
# ## Initialization

# %%%% [code]
import numpy as np
import pandas as pd
from TSbench.TSdata import LoaderTSdf, DatasetOperations

# %%%% [code]
path = "data/example_operations/data"
datatype = "simulated"
split = ["0", "1"]
permission = "overwrite"  # Overwrite is used for repeated execution
loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
loader.restart_dataset()  # fresh re-run

# %%% [markdown]
# ## Data operations

# %%%% [markdown]
# ### Add datatype

# %%%%% [code]
d = {
    "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
    "timestamp": np.arange(10),
    "feature1": np.arange(10, 20),
    "feature2": np.arange(10, 20),
}
df = pd.DataFrame(data=d)
loader.add_data(data=df)
print(loader.df)

# %%%% [markdown]
# ### Add ID

# %%%%% [code]
ID = "added_ID"
d = {
    "timestamp": np.arange(0, 5),
    "feature1": np.arange(5),
    "feature2": np.arange(10, 15),
}
df = pd.DataFrame(data=d)
loader.add_data(df, ID=ID, collision="overwrite")
print(loader.df)  # in memory

# %%%% [markdown]
# ### Add feature

# %%%%% [code]
feature = "added_feature"
d = {"timestamp": np.arange(10), feature: np.arange(10)}
df = pd.DataFrame(data=d)
loader.add_feature(df, ID="added_ID", feature=feature)
print(loader.df)  # in memory

# %%%% [markdown]
# ### Saving changes

# %%%%% [code]
loader.write()

# %%%% [markdown]
# ### Remove data

# %%%%% [code]
empty_loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)
print(empty_loader.df)

# %%%%% [code]
empty_loader.rm_datatype()
assert isinstance(empty_loader.df, pd.DataFrame) and empty_loader.df.size == 0

# %%% [markdown]
# ## Metadata operations

# %%%% [code]
print(loader.metadata)

# %%%% [markdown]
# ### Add metadata

# %%%%% [code]
loader.set_metadata(start=["2016-01-01"])
# loader.add_metadata(start="2016-01-01")
# loader.add_metadata(test=["0", "0"], test2=["1", "1"])

# %%%% [markdown]
# Don't forget to write the changes on the disk

# %%%%% [code]
loader.write()

# %%% [markdown]
# ## Dataset operations
# *Execution order here is important.*

# %%%% [markdown]
# ### Define loaders

# %%%%% [code]
data_path = "data/example_operations/data"
multiprocess_path = "data/example_multiprocessing"
copy_path = "data/example_operations/copy"
move_path = "data/example_operations/move"
merge_path = "data/example_operations/example_merge"
permission = "overwrite"
data_loader = LoaderTSdf(path=data_path, datatype=datatype, permission=permission)
multiprocess_loader = LoaderTSdf(
    path=multiprocess_path, datatype="splitted_data", permission=permission
)
print("Use case metadata")
print("-----------------")
print(data_loader.metadata)
print()
print("Multiprocessing metadata")
print("---------------------")
print(multiprocess_loader.metadata)

# %%%% [markdown]
# ### Remove data from previous executions

# %%%%% [code]
data_loader.set_path(copy_path)
data_loader.rm_dataset()
data_loader.set_path(move_path)
data_loader.rm_dataset()
data_loader.set_path(data_path)

# %%%% [markdown]
# ### Copy the data to `copy_path`

# %%%%% [code]
data_loader.copy_dataset(copy_path)

# %%%% [markdown]
# ### Move data to `move_path`

# %%%%% [code]
data_loader.move_dataset(move_path)

# %%%% [markdown]
# ### Merging dataset

# %%%%% [code]
merge_loader = DatasetOperations.merge_dataset(
    [data_loader, multiprocess_loader], merge_path
)
print("Dataset are merged, here is the metadata")
print(merge_loader.metadata)

# %%% [markdown]
# ## Visualize DataFrame

# %%%% [code]
IDs = ["name1", "added_ID"]
timestamps = ["0", "1"]
dims = ["0"]

print(loader.get_df())
# print(loader.get_df(IDs=IDs, drop=True))
# print(loader.get_df(timestamps=timestamps, drop=True))
# print(loader.get_df(dims=dims, drop=True))
# print(loader.get_df(IDs=IDs, timestamps=timestamps, drop=True))
print(loader.get_df(timestamps=pd.Index(timestamps), dims=dims))
