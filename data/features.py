import numpy as np #arrrays
import pandas as pd #data set structure
import matplotlib.pyplot as plt #visualisation data strructre
import yfinance as yf #package

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from itertools import combinations

from sklearn.feature_selection import mutual_info_regression


TICKER = "AAPL"
df = yf.download(TICKER, '2020-01-01')

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

def add_features(df, TOP_FEATURES, TRAIN_SPLIT):

    df = df.copy()
    df = df.sort_index()

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Avoid division by zero
    close_safe = df["Close"].replace(0, np.nan)
    volume_safe = df["Volume"].replace(0, np.nan)

    # ==========================================================
    # 1–4: PRICE FEATURES
    # ==========================================================

    # Today's opening price relative to yesterday's closing price
    df["Open_Gap"] = (
        df["Open"] / df["Close"].shift(1)
    ) - 1

    # Where the high sits relative to the closing price
    df["High_Close_Ratio"] = (
        df["High"] / close_safe
    ) - 1

    # Where the low sits relative to the closing price
    df["Low_Close_Ratio"] = (
        df["Low"] / close_safe
    ) - 1

    # Intraday trading range as a percentage of closing price
    df["HL_Range"] = (
        df["High"] - df["Low"]
    ) / close_safe

    # ==========================================================
    # 5–7: RETURN FEATURES
    # ==========================================================

    # One-day percentage return
    df["Return_1d"] = df["Close"].pct_change(1)

    # Five-day percentage return
    df["Return_5d"] = df["Close"].pct_change(5)

    # Log return
    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # ==========================================================
    # 8–11: TREND FEATURES
    # ==========================================================

    sma_10 = df["Close"].rolling(window=10).mean()
    sma_20 = df["Close"].rolling(window=20).mean()

    ema_10 = df["Close"].ewm(
        span=10,
        adjust=False
    ).mean()

    ema_20 = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # Price relative to each moving average
    df["SMA_10_Ratio"] = (
        df["Close"] / sma_10
    ) - 1

    df["SMA_20_Ratio"] = (
        df["Close"] / sma_20
    ) - 1

    df["EMA_10_Ratio"] = (
        df["Close"] / ema_10
    ) - 1

    df["EMA_20_Ratio"] = (
        df["Close"] / ema_20
    ) - 1

    # ==========================================================
    # 12: RSI
    # ==========================================================

    price_change = df["Close"].diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    df["RSI_14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # ==========================================================
    # 13–14: MACD
    # ==========================================================

    ema_12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema_12 - ema_26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    macd_histogram = macd - macd_signal

    # Normalise MACD by the current closing price
    df["MACD"] = macd / close_safe
    df["MACD_Histogram"] = macd_histogram / close_safe

    # ==========================================================
    # 15: ATR
    # ==========================================================

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = true_range.rolling(window=14).mean()

    # Normalise ATR by closing price
    df["ATR_14"] = atr_14 / close_safe

    # ==========================================================
    # 16: ROLLING VOLATILITY
    # ==========================================================

    df["Volatility_20"] = (
        df["Return_1d"]
        .rolling(window=20)
        .std()
    )

    # ==========================================================
    # 17: BOLLINGER BAND WIDTH
    # ==========================================================

    rolling_std_20 = (
        df["Close"]
        .rolling(window=20)
        .std()
    )

    upper_band = sma_20 + (2 * rolling_std_20)
    lower_band = sma_20 - (2 * rolling_std_20)

    df["BB_Width"] = (
        upper_band - lower_band
    ) / sma_20.replace(0, np.nan)

    # ==========================================================
    # 18: VOLUME RATIO
    # ==========================================================

    average_volume_20 = (
        df["Volume"]
        .rolling(window=20)
        .mean()
    )

    df["Volume_Ratio_20"] = (
        volume_safe /
        average_volume_20.replace(0, np.nan)
    )

    # ==========================================================
    # 19: VOLUME CHANGE
    # ==========================================================

    df["Volume_Change"] = df["Volume"].pct_change()

    # Large volume spikes can create extreme values
    df["Volume_Change"] = df["Volume_Change"].clip(
        lower=-5,
        upper=5
    )

    # ==========================================================
    # 20: ON-BALANCE VOLUME CHANGE
    # ==========================================================

    direction = np.sign(df["Close"].diff()).fillna(0)

    obv = (
        direction * df["Volume"]
    ).cumsum()

    # Percentage change in OBV is unstable around zero,
    # so use the difference divided by average volume.
    df["OBV_Change"] = (
        obv.diff() /
        average_volume_20.replace(0, np.nan)
    )

    # ==========================================================
    # TARGET
    # ==========================================================

    # Return from today's close to tomorrow's close
    df["Tomorrow_Return"] = (
        df["Close"]
        .pct_change()
        .shift(-1)
    )

    feature_columns = [
        "Open_Gap",
        "High_Close_Ratio",
        "Low_Close_Ratio",
        "HL_Range",
        "Return_1d",
        "Return_5d",
        "Log_Return",
        "SMA_10_Ratio",
        "SMA_20_Ratio",
        "EMA_10_Ratio",
        "EMA_20_Ratio",
        "RSI_14",
        "MACD",
        "MACD_Histogram",
        "ATR_14",
        "Volatility_20",
        "BB_Width",
        "Volume_Ratio_20",
        "Volume_Change",
        "OBV_Change",
    ]

    # Replace infinities created by division
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Only remove rows after every feature and target is created
    df.dropna(
        subset=feature_columns + ["Tomorrow_Return"],
        inplace=True
    )




    X = df[feature_columns]
    y = df["Tomorrow_Return"]

    # train/test split
    train_size = int(TRAIN_SPLIT * len(df))

    X_train_raw = X.iloc[:train_size]
    X_test_raw = X.iloc[train_size:]

    y_train_raw = y.iloc[:train_size]
    y_test_raw = y.iloc[train_size:]

    # mutual information feature selection
    mi = mutual_info_regression(X_train_raw, y_train_raw, random_state=42)

    mi_scores = pd.Series(mi, index=X_train_raw.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    top_features = mi_scores.head(TOP_FEATURES).index.tolist()

    print(f"top features are {top_features}")

    X_train_raw = X_train_raw[top_features]
    X_test_raw = X_test_raw[top_features]


    return df, X_train_raw, X_test_raw, y_train_raw, y_test_raw