import pandas as pd

def moving_average(y: pd.Series, window_n: int = 5) -> pd.Series:
    return y.rolling(window=window_n, min_periods=1).mean()

def persistence(y: pd.Series) -> pd.Series:
    return y.shift(1).bfill()
