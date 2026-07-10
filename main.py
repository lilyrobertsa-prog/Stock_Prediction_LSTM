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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.lstm import PredictionModel


from data.features import add_features


from data.sequences import create_sequences

from train import train_model

from predict import make_predictions

from models.linear_regression import Lin_Regression

from scaling_lstm import lstm_scaler

from evaluate import evaluate_predictions


# settings
#TICKER = "MSFT"
#START_DATE = "2020-01-01"

SEQ_LENGTH = 30
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TOP_FEATURES = 7

def run_experiment(ticker, start_date, end_date, SEQ_LENGTH=SEQ_LENGTH, NUM_EPOCHS=NUM_EPOCHS, HIDDEN_DIM=HIDDEN_DIM, NUM_LAYERS=NUM_LAYERS, DROPOUT=DROPOUT, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE):

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError("No data found. Check ticker and date range.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df, X_train_raw, X_test_raw, y_train_raw, y_test_raw = add_features(df, TOP_FEATURES, TRAIN_SPLIT = TRAIN_SPLIT)


    ##linear regression

    linear = Lin_Regression()

    linear.train(X_train_raw, y_train_raw)

    y_train_pred_lin = linear.predict(X_train_raw)
    y_test_pred_lin = linear.predict(X_test_raw)

    linear_pred = linear.predict(X_test_raw)

    train_rmse_lin, test_rmse_lin, direction_accuracy_lin, abs_error_lin = evaluate_predictions(y_train_pred_lin, y_test_pred_lin, y_train_raw, y_test_raw)
    


    ##


    ##LSTM

    X_train_scaled, X_test_scaled, y_test_scaled, y_train_scaled, target_scaler = lstm_scaler(X_train_raw, X_test_raw, y_train_raw, y_test_raw)

    X_train_tensor, y_train_tensor = create_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)

    X_test_tensor, y_test_tensor = create_sequences(X_test_scaled, y_test_scaled, SEQ_LENGTH)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PredictionModel(
            input_dim=X_train_raw.shape[1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_dim=1,
            dropout=DROPOUT
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS
        )

    y_train_pred, y_test_pred, y_train_actual, y_test_actual = make_predictions(
            model=model,
            X_train=X_train_tensor,
            X_test=X_test_tensor,
            y_train=y_train_tensor,
            y_test=y_test_tensor,
            target_scaler=target_scaler,
            device=device
        )


    train_rmse_l, test_rmse_l, direction_accuracy_l, abs_error_l = evaluate_predictions(y_train_pred, y_test_pred, y_train_actual, y_test_actual)

    ##

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "linear_train_rmse": round(train_rmse_lin, 6),
        "linear_test_rmse": round(test_rmse_lin, 6),
        "linear_direction_accuracy": round(direction_accuracy_lin, 4),
        "abs_error_lin": round(abs_error_lin, 6),
        "lstm_train_rmse": round(train_rmse_l, 6),
        "lstm_test_rmse": round(test_rmse_l, 6),
        "lstm_direction_accuracy": round(direction_accuracy_l, 4),
        "abs_error_l": round(abs_error_l, 6),
        "seq_length": SEQ_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "features_used": list(X_train_raw.columns)
    }