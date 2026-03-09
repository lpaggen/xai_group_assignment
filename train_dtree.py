from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from itertools import product
import json


with open("nn_gridsearch_values.json", "r") as f:  # load constants for tree
    grid = json.load(f)["tree"]  # constants for all methods are in the same file


def _train_and_eval_tree(
    X_train,
    X_test,
    y_train,
    y_test,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    criterion,
):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    preds = (probs >= 0.5).astype(int)
    return auc, preds


def gridsearch_tree(X_train, X_test, y_train, y_test):
    results = []
    for max_depth, min_split, min_leaf, criterion in product(
        grid["max_depth"],
        grid["min_samples_split"],
        grid["min_samples_leaf"],
        grid["criterion"],
    ):
        if max_depth == 0:  # because JSON can't encode Null, None
            max_depth = None
        auc, preds = _train_and_eval_tree(
            X_train, X_test, y_train, y_test, max_depth, min_split, min_leaf, criterion
        )
        results.append(
            {
                "max_depth": max_depth,
                "min_samples_split": min_split,
                "min_samples_leaf": min_leaf,
                "criterion": criterion,
                "auc": auc,
                "predictions": preds,
            }
        )
    best = max(results, key=lambda x: x["auc"])
    print(f"Best tree config: {best}")
    return best
