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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)  # working on my laptop, use CUDA if you have nvidia gpu, CPU would do fine also, small models

with open("nn_gridsearch_values.json", "r") as f:  # load constants for NN
    grid = json.load(f)["nn"]  # constants for all methods are in the same file


def _train_and_eval_nn_gridsearch(
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
        auc = roc_auc_score(y_test, test_preds)
        return auc, test_labels


def gridsearch_nn(
    input_dim: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 30,
) -> dict:
    """
    Trains N (cartesian product of how many values set in the JSON grid) neural networks and outputs the best model wrt accuracy
    """

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
        result_tuple = _train_and_eval_nn_gridsearch(
            X_train,
            X_test,
            y_train,
            y_test,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        auc = result_tuple[0]
        predictions = result_tuple[1]
        results.append(
            {
                "lr": learn_rate,
                "batch_size": batch_size,
                "optimizer": opt_name,
                "architecture": architecture,
                "dropout": dropout,
                "epochs": n_epochs,
                "auc": auc,
                "predictions": predictions,
            }
        )
    best = max(results, key=lambda x: x["auc"])
    print(f"Best config: {best}")
    return best


def run_xai(
    best: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> None:
    model = CreditNN(
        input_dim=X_train.shape[1],
        hidden_layers=best["architecture"],
        dropout=best["dropout"],
    ).to(device)

    optimizer = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
    }[best["optimizer"]](model.parameters(), lr=best["lr"])

    _train_and_eval_nn_gridsearch(
        X_train,
        X_test,
        y_train,
        y_test,
        model=model,
        optimizer=optimizer,
        batch_size=best["batch_size"],
        n_epochs=best["epochs"],
    )
    model.eval()

    def predict_proba(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(device)
            probs = model(t).squeeze().cpu().numpy()
        probs = np.atleast_1d(probs)
        return np.column_stack([1 - probs, probs])

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["bad", "good"],
        mode="classification",
        random_state=42
    )
    exp = explainer_lime.explain_instance(X_test[0], predict_proba, num_features=10)
    exp.save_to_file("lime_instance.html")
    print("Saved lime_instance.html")
    print("\nLIME top features for instance 0:")
    for feat, weight in exp.as_list():
        print(f"  {feat}: {weight:.4f}")

    print("Computing SHAP values...")
    background = X_train[:500]
    explainer_shap = shap.KernelExplainer(lambda x: predict_proba(x)[:, 1], background)
    shap_values = explainer_shap.shap_values(X_test[:50])
    shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")
