import math
from .ML.model_training import train_knn
from .ML.utils import read_table
import re


df = read_table("datasets/features_table.csv")
df = df.reset_index(drop=True)

X = df.drop(
    columns=["patient_index", "video_index", "arousal", "valence", "Unnamed: 0"]
)
X = X.drop(columns=X.columns[X.columns.str.contains('delta', case=False, na=False)])
X = X.sort_index(axis=1)

arousal_target = df["arousal"].astype(float)
valence_target = df["valence"].astype(float)

best_model = None
best_mse = math.inf
best_n = 1
arousal_model, X_test, y_test = train_knn(
    X, None, arousal_target, None, neighbors=best_n
)
valence_model, X_test, y_test = train_knn(
    X, None, valence_target, None, neighbors=best_n
)
