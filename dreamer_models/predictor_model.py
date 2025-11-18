import math
import numpy as np
import pandas as pd
from .ML.model_training import train_lstm
from .ML.utils import read_table, filter_features
from .ML.labels import relabel_target_from_video_map
import re


df = read_table("datasets/features_table_psd_only.csv")
df = df.reset_index(drop=True)

exclude_users = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 21, 22]
if exclude_users:
    df = df[~df["patient_index"].isin(exclude_users)].reset_index(
        drop=True
    )
# df = relabel_target_from_video_map(df)
X = df.drop(
    columns=["patient_index", "video_index", "arousal", "valence", "Unnamed: 0"]
)
X = X.sort_index(axis=1)

# TODO: select subjects and training set

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


X = X.loc[:, features]
print(X.shape)

best_model = None
best_mse = math.inf
arousal_model, X_test_eval, y_test_eval = train_lstm(
    X,
    None,
    arousal_target,
    None,
    lr=0.0001,
    epochs=10,
    units=512,
    batch_size=256,
    bidirectional=False,
)
valence_model, X_test, y_test = train_lstm(
    X,
    None,
    valence_target,
    None,
    lr=0.0001,
    epochs=10,
    units=512,
    batch_size=256,
    bidirectional=False,
)
