import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def split_x_y(
    df: pd.DataFrame,
    ) -> tuple[
    np.ndarray
    ]:  # pass credit df, get train-test split, reuse in other parts of code
    df.columns = df.columns.str.strip("'")
    y = df["class"]
    X = df.drop(columns=["class"])
    y = (y == "good").astype(float)
    X = pd.get_dummies(X).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42  # 80/20 split
    )
    scaler = StandardScaler()
    X_train: np.ndarray = scaler.fit_transform(X_train)
    X_test: np.ndarray = scaler.transform(X_test)
    return y_train, y_test, X_train, X_test
