
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def lstm_scaler(
    X_train_raw,
    X_test_raw,
    y_train_raw,
    y_test_raw
):
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    X_test_scaled = feature_scaler.transform(X_test_raw)

    y_train_scaled = target_scaler.fit_transform(y_train_raw.to_frame())
    y_test_scaled = target_scaler.transform(y_test_raw.to_frame())

    return X_train_scaled, X_test_scaled, y_test_scaled, y_train_scaled, target_scaler

