import pandas as pd
import joblib

def standard_deviation(df):
    """入力データを標準化する関数

    Parameters
    ----------
    df : pandas.df
        pandasのDataFrame型の入力データ

    Returns
    -------
    df
        標準化されたpandasのDataFrame
    """
    return df.std(ddof=0)

