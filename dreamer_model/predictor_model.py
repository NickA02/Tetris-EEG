from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import math
from .ML.model_training import train_knn
from .ML.utils import read_table
import re

df = read_table("datasets/features_table.csv")
df = df.reset_index(drop=True)


### TODO: FULL FEATURES
X = df.drop(columns=["patient_index", "video_index", "arousal", "valence"])
bands = ["theta", "alpha", "betaL", "betaH", "gamma"]
pattern = rf"^[A-Za-z0-9]+_({'|'.join(map(re.escape, bands))})$"
X = X.loc[:, X.columns.str.fullmatch(pattern, case=False)]

arousal_target = df["arousal"].astype(float)
valence_target = df["valence"].astype(float)

best_model = None
best_mse = math.inf
best_n = 1
arousal_model, X_test, y_test = train_knn(X, None, arousal_target, None, neighbors=21)
valence_model, X_test, y_test = train_knn(X, None, valence_target, None, neighbors=21)