# Stock Returns Prediction using LSTM and Linear Regression

## Overview

This project predicts next-day stock returns using historical market data and technical indicators, and produces metrics to evaluate the accuracy of each model.

The application is built with Python, PyTorch and Flask, allowing users to enter any stock ticker and compare the performance of an LSTM neural network against a Linear Regression baseline.

## Features

- Downloads and caches historical stock market data using yfinance, automatically updating local datasets with newly available trading data to minimise repeated downloads.
- Caches trained LSTM models based on ticker symbol, date range, and model configuration, allowing previously trained models to be reused without retraining.
- Generate technical indicators automatically
- LSTM prediction model
- Linear Regression baseline
- Compare multiple evaluation metrics
- Flask web interface
- Plots of predictions
- Error handling


## Models

### Linear Regression
- Fast baseline
- Easy to interpret

### LSTM
- Learns temporal patterns
- Uses sliding window sequences

## Evaluation

Models are compared using:

- RMSE
- Mean Absolute Error
- Directional Accuracy

## Technologies

- Python
- PyTorch
- scikit-learn
- Flask
- pandas
- NumPy
- matplotlib
- yfinance


## Usage

python app.py


## Example

![alt text](image-1.png)

## Project Structure

project/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── scaling_lstm.py
├── plots.py
├── main.py
├── models/
├── data/
├── saved_data/
├── saved_models/
├── templates/
└── README.md

## Future Improvements

- Transformer model
- Hyperparameter optimisation
- Model checkpoint saving
- Docker deployment
- Live predictions
- Deploy on Render
- Add SQL database
- Make option to retrain model over same data



Let the user choose a supported ticker, such as AAPL or MSFT.
Find the saved model or saved results for that ticker.
Load them from disk.
Display the model’s metrics, selected features and graphs.
Optionally use the saved model to make predictions on newly downloaded data.
Return the page quickly without using large amounts of memory.


add saved_results to save the metrics in results:     results = {
        "ticker": ticker,
        "linear_test_rmse": float(test_rmse_lin),
        "lstm_test_rmse": float(test_rmse_l),
        "linear_test_mae": float(abs_error_lin),
        "lstm_test_mae": float(abs_error_l),
        "linear_direction": float(direction_accuracy_lin),
        "lstm_direction": float(direction_accuracy_l),
        "sequence_length": int(SEQ_LENGTH),
        "hidden_dim": int(HIDDEN_DIM),
        "epochs": int(NUM_EPOCHS)
    }

    