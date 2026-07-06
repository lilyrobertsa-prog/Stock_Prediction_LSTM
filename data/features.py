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

    df["Return_1d"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA10_ratio"] = df["Close"] / df["MA_10"]
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df["Tomorrow_Return"] = df["Close"].pct_change().shift(-1)

    df = df.dropna()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = df.copy()

    # choose starting features
    features = [
        "Close",
        "Volume",
        "Return_1d",
        "Volatility_20",
        "MA10_ratio",
        "HL_Range",
        "Volume_Ratio"
    ]

    X = data[features]
    y = data["Tomorrow_Return"]

    # train/test split
    train_size = int(TRAIN_SPLIT * len(data))

    X_train_raw = X.iloc[:train_size]
    X_test_raw = X.iloc[train_size:]

    y_train_raw = y.iloc[:train_size]
    y_test_raw = y.iloc[train_size:]

    # mutual information feature selection
    #mi = mutual_info_regression(X_train_raw, y_train_raw, random_state=42)

    #mi_scores = pd.Series(mi, index=X_train_raw.columns)
    #mi_scores = mi_scores.sort_values(ascending=False)

    top_features = features

    X_train_raw = X_train_raw[top_features]
    X_test_raw = X_test_raw[top_features]

    return df, X_train_raw, X_test_raw, y_train_raw, y_test_raw