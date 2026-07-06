import numpy as np
import torch

def create_sequences(X, y, seq_length):

    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])

    return (
        torch.tensor(np.array(X_seq), dtype=torch.float32),
        torch.tensor(np.array(y_seq), dtype=torch.float32).view(-1, 1)
    )