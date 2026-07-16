import sqlite3
from pathlib import Path

import json
import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parent / "stock_models.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def initialise_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,

                ticker TEXT NOT NULL,
                model_type TEXT NOT NULL,

                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,

                sequence_length INTEGER,
                hidden_dimension INTEGER,
                number_of_layers INTEGER,
                dropout REAL,
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                top_features INTEGER,

                selected_features TEXT,

                train_rmse REAL,
                test_rmse REAL,
                test_mae REAL,
                directional_accuracy REAL,

                mean_prediction REAL,
                mean_actual REAL,
                prediction_std REAL,
                actual_std REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_experiment(
    ticker,
    model_type,
    start_date,
    end_date,
    sequence_length,
    hidden_dimension,
    number_of_layers,
    dropout,
    epochs,
    batch_size,
    learning_rate,
    top_features,
    selected_features,
    train_rmse,
    test_rmse,
    test_mae,
    directional_accuracy,
    mean_prediction,
    mean_actual,
    prediction_std,
    actual_std,
):
    selected_features_json = json.dumps(list(selected_features))

    start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                ticker,
                model_type,
                start_date,
                end_date,
                sequence_length,
                hidden_dimension,
                number_of_layers,
                dropout,
                epochs,
                batch_size,
                learning_rate,
                top_features,
                selected_features,
                train_rmse,
                test_rmse,
                test_mae,
                directional_accuracy,
                mean_prediction,
                mean_actual,
                prediction_std,
                actual_std
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                ticker.upper(),
                model_type,
                str(start_date),
                str(end_date),
                sequence_length,
                hidden_dimension,
                number_of_layers,
                dropout,
                epochs,
                batch_size,
                learning_rate,
                top_features,
                selected_features_json,
                float(train_rmse),
                float(test_rmse),
                float(test_mae),
                float(directional_accuracy),
                float(mean_prediction),
                float(mean_actual),
                float(prediction_std),
                float(actual_std),
            ),
        )

        return cursor.lastrowid
    
def get_experiments(ticker=None, model_type=None):
    query = """
        SELECT *
        FROM experiments
        WHERE 1 = 1
    """

    parameters = []

    if ticker:
        query += " AND ticker = ?"
        parameters.append(ticker.upper())

    if model_type:
        query += " AND model_type = ?"
        parameters.append(model_type)

    query += " ORDER BY created_at DESC"

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]