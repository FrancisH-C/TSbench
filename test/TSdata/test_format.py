# initialization
import numpy as np
import pandas as pd
from TSbench.TSdata import TSloader, DataFormat

# generated data
ID = "ABC"
dim_label = ["0", "1"]
returns = np.array([[0, 1], [2, 3], [4, 5]])
vol = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])

df1 = DataFormat.np_to_TSdf(returns, ID=ID, dim_label=dim_label, feature="returns")
df1 = DataFormat.np_to_TSdf(vol, df=df1, ID=ID, dim_label=dim_label, feature="vol")
# d = {"returns": returns, "vol": vol}
# df2 = DataFormat.dict_to_TSdf(d, ID=ID, dim_label=dim_label)
# assert df1.equals(df2)
