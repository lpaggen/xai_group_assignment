import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import lime
import lime.lime_tabular
from neuralnet import CreditNN
from preprocess_data import train_test_split
import numpy as np
from itertools import product
import json


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)  # working on my laptop, use CUDA if you have to

with open("nn_gridsearch_values.json", "r") as f:  # load constants for NN
    grid = json.load(f)


def train_and_eval_nn_gridsearch(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model: CreditNN,
    optimizer: torch.optim,
    batch_size: int,
    n_epochs: int,
) -> tuple[float, np.ndarray]:
    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to(device)
    y_test_t = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
    )
    # input_dim = X_train.shape[1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs} — Loss: {epoch_loss/len(train_loader):.4f}"
            )

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).squeeze().cpu().numpy()
        test_labels = (test_preds >= 0.5).astype(int)
        accuracy = (test_labels == y_test).mean()
        return accuracy, test_labels  # tuple, careful in next function


def gridsearch_nn(
    input_dim: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 30,
) -> None:
    results = (
        []
    )  # save results for each run, compare later take max accuracy on test set
    for learn_rate, batch_size, opt_name, architecture, dropout, n_epochs in product(
        grid["learning_rate"],
        grid["batch_size"],
        grid["optimizer"],
        grid["architecture"],
        grid["dropout"],
        grid["epochs"],
    ):
        model = CreditNN(
            input_dim=input_dim, hidden_layers=architecture, dropout=dropout
        ).to(device)
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        optimizer = optimizers[opt_name](model.parameters(), lr=learn_rate)
        result_tuple = train_and_eval_nn_gridsearch(
            X_train,
            X_test,
            y_train,
            y_test,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        acc = result_tuple[0]
        predictions = result_tuple[1]
        results.append(
            {
                "lr": learn_rate,
                "batch_size": batch_size,
                "optimizer": opt_name,
                "architecture": architecture,
                "dropout": dropout,
                "epochs": n_epochs,
                "accuracy": acc,
                "predictions": predictions
            }
        )
    best = max(results, key=lambda x: x["accuracy"])
    print(f"Best config: {best}")
    return results
