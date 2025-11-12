import math
import numpy as np
import pandas as pd
from .ML.model_training import train_lstm
from .ML.utils import read_table, filter_features
import re


df = read_table("datasets/features_table.csv")
df = df.reset_index(drop=True)

X = df.drop(
    columns=["patient_index", "video_index", "arousal", "valence", "Unnamed: 0"]
)
X = X.drop(columns=X.columns[X.columns.str.contains("delta", case=False, na=False)])
X = X.sort_index(axis=1)

arousal_target = df["arousal"].astype(float)
valence_target = df["valence"].astype(float)

features = filter_features(X.columns, remove_bands=["gamma", "delta"])
arousal_target = pd.Series(
    np.where(arousal_target > 3.8, "high", "low"),
    index=arousal_target.index,
    dtype="string",
)
valence_target = pd.Series(
    np.where(valence_target > 3.8, "high", "low"),
    index=valence_target.index,
    dtype="string",
)


def balance(X, y, seed=5):
    c = y.value_counts()
    if c.get("high", 0) == c.get("low", 0):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    maj = c.idxmax()
    m = c.min()
    keep = y[y != maj].index.union(y[y == maj].sample(m, random_state=seed).index)
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


X, arousal_target = balance(X, arousal_target, seed=5)
print("arousal_train counts:\n", arousal_target.value_counts(dropna=False))
X, valence_target = balance(X, arousal_target, seed=5)
print("arousal_train counts:\n", arousal_target.value_counts(dropna=False))



best_model = None
best_mse = math.inf
arousal_model, X_test_eval, y_test_eval = train_lstm(
    X,
    None,
    arousal_target,
    None,
    lr=0.001,
    epochs=10,
    units=1024,
    batch_size=1024,
    bidirectional=False,
)
valence_model, X_test, y_test = train_lstm(
    X,
    None,
    valence_target,
    None,
    lr=0.001,
    epochs=10,
    units=1024,
    batch_size=1024,
    bidirectional=False,
)
