import pandas as pd
from TSbench.TSdata.DatasetOperations import InitialFormatCSV


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df["ID"] = "SPY"
    df = df.rename(columns={"Date": "timestamp"})
    df = df.sort_values(by="timestamp")
    return df


def split_from_filename(_filename: str) -> str:
    return "1"


if __name__ == "__main__":
    pre_process_path = "data/SPY/original/"
    data_path = "data/SPY/input/"
    datatype = "stock"

    dataset_format = InitialFormatCSV(
        datatype=datatype,
        original_path=pre_process_path,
        processed_path=data_path,
        process_df=process_df,
        split_from_filename=split_from_filename,
        test=False,
    )
    dataset_format.process()
