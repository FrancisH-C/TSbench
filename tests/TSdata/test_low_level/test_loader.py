import numpy as np
import pandas as pd
import pytest

from TSbench.TSdata import DatasetOperations, LoaderTSdf, LoaderTSdfCSV


def same_data(df1, df2, debug=False):
    """Verify if two DataFrame have the same data

    DataFrame.equals does not work, because it assumes index and
    columns position is constant.
    """
    df1 = df1.fillna(-1)
    df2 = df2.fillna(-1)

    if debug:
        print(df1)
        print(df2)

    if df1.shape != df2.shape:
        return False

    for index in df1.index:
        for feature in df1.columns:
            x = df1.loc[index][feature]
            y = df2.loc[index][feature]
            if x != y:
                if debug:
                    print("Not equal")
                    print("index =", index)
                    print("feature =", feature)
                    print("df1 element:")
                    print(df1.loc[index][feature])
                return False
    return True


def simple_loader():
    """Simple loader for test."""
    path = "data/test_add"
    datatype = "simulated"
    permission = "overwrite"
    loader = LoaderTSdf(path=path, datatype=datatype, permission=permission)

    ID = "added_ID"
    feature = "added_feature"

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
    }
    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}
    df = pd.DataFrame(data=d)
    df_feature = pd.DataFrame(data=d_feature)

    loader.set_df(df=df.copy())
    df["ID"] = ID
    loader.add_data(df, ID=ID, collision="overwrite")
    loader.add_feature(df_feature, ID=ID, feature=feature)

    return loader


def test_loaderCSV():
    """Simple CSV loader."""
    path = "data/test_CSV"
    datatype = "simulated"
    permission = "overwrite"
    loader = LoaderTSdfCSV(path=path, datatype=datatype, permission=permission)

    ID = "added_ID"
    feature = "added_feature"

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
    }
    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}
    df = pd.DataFrame(data=d)
    df_feature = pd.DataFrame(data=d_feature)

    loader.set_df(df=df.copy())
    df["ID"] = ID
    loader.add_data(df, ID=ID, collision="overwrite")
    loader.add_feature(df_feature, ID=ID, feature=feature)

    loader.write()
    loader.load()


def test_permission():
    testpath = "data/test_nonexistent"
    loader = simple_loader()

    loader.set_permission("read")
    # Dataset operations
    with pytest.raises(ValueError):
        loader.copy_dataset(testpath)
    with pytest.raises(ValueError):
        loader.write()
    with pytest.raises(ValueError):
        loader.write_metadata()
    with pytest.raises(ValueError):
        loader.merge_metadata()
    with pytest.raises(ValueError):
        loader.rm_dataset()
    with pytest.raises(ValueError):
        loader.move_dataset(testpath)

    # Overwrite operations
    loader.set_permission("write")
    ID = "added_ID"
    feature = "feature0"
    df = pd.DataFrame(index=pd.Index(["ID", "timestamp"]), columns=pd.Index([feature]))

    with pytest.raises(ValueError):
        loader.rm_ID(ID)
    with pytest.raises(ValueError):
        loader.rm_feature(feature)
    with pytest.raises(ValueError):
        loader.set_df(pd.DataFrame(index=pd.Index(["ID", "timestamp"])))

    with pytest.raises(ValueError):
        loader.add_data(df.copy(), ID=ID, collision="overwrite")
    with pytest.raises(ValueError):
        loader.add_feature(df.copy(), ID=ID, feature=feature)


def test_dataset_operations():
    path = "data/test_dataset_operations/"
    copy_path = "data/test_dataset_operations/copy/"
    move_path = "data/test_dataset_operations/move/"
    merge_path = "data/test_dataset_operations/merge/"

    loader = LoaderTSdf(path=path, datatype="test")
    loader_other = LoaderTSdf(path=path, datatype="test")

    loader.rm_dataset()
    loader._create_path()
    loader.write()

    loader.move_dataset(move_path)
    loader.copy_dataset(copy_path)
    loader_other.set_path(copy_path)
    DatasetOperations.merge_dataset([loader, loader_other], merge_path)


def test_add_data():
    """Test add instructions."""
    loader = simple_loader()
    solution_df = loader.df.copy()

    # add_data
    ID = "added_ID"
    feature = "added_feature"

    d = {
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
        "added_feature": np.hstack((list(range(15, 19)), np.full(6, np.nan))),
    }
    df = pd.DataFrame(data=d)  # added_ID DataFrame

    # ignore and overwrite
    d_ID = {
        "timestamp": list(map(str, range(0, 4))),
        "feature0": list(range(4)),
        "feature1": list(range(10, 14)),
    }
    df_ID = pd.DataFrame(data=d_ID)  # DataFrame with different data for ID

    loader.add_data(df_ID.copy(), ID=ID, collision="ignore")
    assert same_data(loader.df, solution_df)

    df["ID"] = ID
    loader.add_data(df.copy(), ID=ID, collision="overwrite")
    assert same_data(loader.df, solution_df)

    # append and update
    d1 = {
        "timestamp": list(map(str, range(0, 4))),
        "feature0": list(range(4)),
        "feature1": list(range(10, 14)),
        "added_feature": list(range(15, 19)),
    }
    d2 = {
        "timestamp": list(map(str, range(4, 10))),
        "feature0": list(range(4, 10)),
        "feature1": list(range(14, 20)),
        "added_feature": np.full(6, np.nan),
    }
    df1 = pd.DataFrame(data=d1)  # first half of the data
    df2 = pd.DataFrame(data=d2)  # second half of the data

    loader.add_data(df1.copy(), ID=ID, collision="overwrite")
    loader.add_data(df2.copy(), ID=ID, collision="update")
    assert same_data(loader.df, solution_df)

    # add_feature
    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}
    d_feature_other = {
        "timestamp": list(map(str, range(4))),
        feature: list(range(15, 19)),
    }
    df_feature = pd.DataFrame(data=d_feature)  # feature DataFrame
    df_feature_other = pd.DataFrame(
        data=d_feature_other
    )  # DataFrame with different data for feature

    loader.add_feature(df_feature_other.copy(), ID=ID, feature=feature)
    loader.add_feature(df_feature_other.copy(), ID=ID, feature=feature)
    assert same_data(loader.df, solution_df)
    loader.add_feature(df_feature.copy(), ID=ID, feature=feature)
    assert same_data(loader.df, solution_df)


def test_rm_data():
    """Test add instructions."""
    loader = simple_loader()
    ID = "added_ID"
    feature = "added_feature"

    # Initially present
    assert ID in loader.df.index
    assert feature in loader.df.columns

    # removed
    loader.rm_ID(ID)
    assert ID not in loader.df.index
    loader.rm_feature(feature)
    assert feature not in loader.df.columns


def test_metadata_operations():
    """Test with metadata."""
    loader = simple_loader()

    # add
    # list or no list input
    loader.append_to_metadata(test_metadata=[1])
    assert loader.metadata["test_metadata"].iloc[0] == [1]
    loader.append_to_metadata(test_metadata=[1])
    assert loader.metadata["test_metadata"].iloc[0] == [1]
    # different value add
    loader.append_to_metadata(test_metadata=[2])
    loader.append_to_metadata(test_metadata=[3])
    assert (loader.metadata["test_metadata"].iloc[0] == [1, 2, 3]).all()

    # overwrite
    # list or no list input
    loader.set_metadata(test_metadata=[1])
    assert loader.metadata["test_metadata"].iloc[0] == [1]
    loader.set_metadata(test_metadata=[1])
    assert loader.metadata["test_metadata"].iloc[0] == [1]

    ## set datatype to call all the metadata initialization
    loader.set_datatype("test")


def test_complex_interactions():
    """Test interactions."""
    loader = simple_loader()
    solution_df = loader.df.copy()
    ID = "added_ID"
    feature = "added_feature"

    d_ID = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
        "added_feature": np.hstack((list(range(15, 19)), np.full(6, np.nan))),
    }

    d1 = {
        "timestamp": list(map(str, range(0, 5))),
        "feature0": list(range(5)),
        "feature1": list(range(10, 15)),
    }

    d2 = {
        "timestamp": list(map(str, range(5, 10))),
        "feature0": list(range(5, 10)),
        "feature1": list(range(15, 20)),
    }

    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}

    df_ID = pd.DataFrame(data=d_ID)  # added_ID DataFrame
    df1 = pd.DataFrame(data=d1)  # name1 DataFrame
    df2 = pd.DataFrame(data=d2)  # name2 DataFrame
    df_feature = pd.DataFrame(data=d_feature)  # added_feature DataFrame

    loader.add_data(df_ID.copy(), ID=ID, collision="overwrite")
    loader.add_data(df1.copy(), ID="name1", collision="overwrite")
    loader.add_data(df2.copy(), ID="name2", collision="overwrite")
    assert same_data(loader.df.loc["name1"], solution_df.loc["name1"])
    assert same_data(loader.df.loc["name2"], solution_df.loc["name2"])

    loader.add_data(df1.copy(), ID="name2", collision="update")
    loader.add_data(df2.copy(), ID="name1", collision="update")
    loader.add_feature(df_feature.copy(), ID="name1", feature=feature)
    loader.add_feature(df_feature.copy(), ID="name2", feature=feature)
    assert same_data(loader.df.loc["name1"], solution_df.loc["added_ID"])
    assert same_data(loader.df.loc["name2"], solution_df.loc["added_ID"])
