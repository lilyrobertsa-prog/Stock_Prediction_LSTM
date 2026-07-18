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

import os

from database import save_experiment

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



def load_or_update_stock_data(ticker):

    """
    Load a ticker's cached daily history and update it with any newer data.

    Returns:
        A DataFrame containing all cached available daily data.
    """
    ticker = ticker.upper().strip()

    os.makedirs("saved_data", exist_ok=True)

    data_path = os.path.join(
        "saved_data",
        f"{ticker}.csv"
    )

    if not os.path.exists(data_path):
        print(f"Downloading all available data for {ticker}...")

        df = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise ValueError(
                "No data found. Check the ticker symbol."
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        df.to_csv(data_path)

        return df

    print(f"Loading saved data for {ticker}...")

    saved_df = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=True
    )

    saved_df.index = pd.to_datetime(saved_df.index)

    last_saved_date = saved_df.index.max()

    # Start from the last saved date so that the final stored row
    # can also be refreshed if Yahoo has corrected it.
    update_start = last_saved_date.strftime("%Y-%m-%d")

    print(
        f"Checking for updates from {update_start}..."
    )

    new_df = yf.download(
        ticker,
        start=update_start,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if not new_df.empty:
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)

        new_df.index = pd.to_datetime(new_df.index)

        combined_df = pd.concat(
            [saved_df, new_df]
        )

        # If the same date appears twice, keep the newly downloaded row.
        combined_df = combined_df[
            ~combined_df.index.duplicated(keep="last")
        ]

        combined_df = combined_df.sort_index()
        combined_df.index.name = "Date"

        combined_df.to_csv(data_path)

        print(
            f"Data updated. Latest date: "
            f"{combined_df.index.max().date()}"
        )

        return combined_df

    print("Saved data is already up to date.")

    return saved_df








def run_experiment(ticker, start_date, end_date, SEQ_LENGTH=SEQ_LENGTH, NUM_EPOCHS=NUM_EPOCHS, HIDDEN_DIM=HIDDEN_DIM, NUM_LAYERS=NUM_LAYERS, DROPOUT=DROPOUT, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE):

    os.makedirs("saved_models", exist_ok=True)

    model_path = os.path.join(
        "saved_models",
        f"{ticker}_"
        f"{start_date}_"
        f"{end_date}_"
        f"seq{SEQ_LENGTH}_"
        f"h{HIDDEN_DIM}_"
        f"layers{NUM_LAYERS}.pth"
    )

    df_all = load_or_update_stock_data(ticker)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df = df_all.loc[
        (df_all.index >= start_date) &
        (df_all.index < end_date)
    ].copy()

    if start_date >= end_date:
        raise ValueError("The start date must be before the end date.")

    if df.empty:
        raise ValueError(
            "No data found for the selected date range."
        )
    
    
    if len(df) < 100:
        raise ValueError("Not enough historical data for this model.")
    
    if len(df) <= SEQ_LENGTH:
        raise ValueError("The selected date range is too short")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df, X_train_raw, X_test_raw, y_train_raw, y_test_raw = add_features(df, TOP_FEATURES, TRAIN_SPLIT = TRAIN_SPLIT)


    ##linear regression

    linear = Lin_Regression()

    linear.train(X_train_raw, y_train_raw)

    y_train_pred_lin = linear.predict(X_train_raw)

    y_test_pred_lin = linear.predict(X_test_raw)

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

    print("Model path:", os.path.abspath(model_path))
    print("Model exists:", os.path.exists(model_path))

    if os.path.exists(model_path):

        print("Loading saved model...")

        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

    else:

        print("Training model...")

        model = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=NUM_EPOCHS
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        #saves the models parameters, not the model itself

    y_train_pred, y_test_pred, y_train_actual, y_test_actual = make_predictions(
            model=model,
            X_train=X_train_tensor,
            X_test=X_test_tensor,
            y_train=y_train_tensor,
            y_test=y_test_tensor,
            target_scaler=target_scaler,
            device=device
        )
    
    print(y_test_pred[:10])
    print(y_test_actual[:10])


    train_rmse_l, test_rmse_l, direction_accuracy_l, abs_error_l = evaluate_predictions(y_train_pred, y_test_pred, y_train_actual, y_test_actual)

    ##
    #y_test_actual == y_test_raw

    save_experiment(
        ticker=ticker,
        model_type="Linear Regression",
        start_date=start_date,
        end_date=end_date,

        sequence_length=None,
        hidden_dimension=None,
        number_of_layers=None,
        dropout=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,

        top_features=TOP_FEATURES,
        selected_features=list(X_train_raw.columns),

        train_rmse=train_rmse_lin,
        test_rmse=test_rmse_lin,
        test_mae=abs_error_lin,
        directional_accuracy=direction_accuracy_lin,

        mean_prediction=np.mean(y_test_pred_lin),
        mean_actual=np.mean(y_test_raw),
        prediction_std=np.std(y_test_pred_lin),
        actual_std=np.std(y_test_raw),
    )

    save_experiment(
        ticker=ticker,
        model_type="LSTM",
        start_date=start_date,
        end_date=end_date,

        sequence_length=SEQ_LENGTH,
        hidden_dimension=HIDDEN_DIM,
        number_of_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,

        top_features=TOP_FEATURES,
        selected_features=list(X_train_raw.columns),

        train_rmse=train_rmse_l,
        test_rmse=test_rmse_l,
        test_mae=abs_error_l,
        directional_accuracy=direction_accuracy_l,

        mean_prediction=np.mean(y_test_pred),
        mean_actual=np.mean(y_test_raw),
        prediction_std=np.std(y_test_pred),
        actual_std=np.std(y_test_raw),
    )


    results = {
        "y_test_pred_lin": y_test_pred_lin,
        "y_test_raw": np.asarray(y_test_raw).reshape(-1),

        "y_test_pred_lstm": np.asarray(y_test_pred).reshape(-1),

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
        "selected_features": list(X_train_raw.columns)
    }


    return results

