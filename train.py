import torch

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):

    model.to(device)

    for epoch in range(num_epochs):

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #if epoch % 10 == 0:
            #print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model