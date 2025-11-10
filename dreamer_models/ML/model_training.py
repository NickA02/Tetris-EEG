import pandas as pd
import numpy as np
from .utils import read_table
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import random


def random_train_test_split(
    test_size: float = 0.1,
    target: str = "arousal",
    shuffle_random_state: int | None = None,
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)
    df = df_main.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe columns.")

    y = df[target].astype(float)

    blacklist = {
        "patient_index",
        "video_index",
        "arousal",
        "valence",
        "Unnamed: 0",
        target,
    }

    X = df.drop(columns=[c for c in blacklist if c in df.columns], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=shuffle_random_state,
        stratify=None,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def omit_patient_video(  # PERFORMS LOO (Leave-one-trial-out)
    target: str = "arousal", random_state: int | None = None, trials=1
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)
    df_main = df_main.drop(columns=["Unnamed: 0"], errors="ignore")

    X = df_main.drop(
        columns=["patient_index", "video_index", "arousal", "valence"], errors="ignore"
    )
    y = df_main[target]

    trial_id = pd.Series(
        list(map(tuple, df_main[["patient_index", "video_index"]].to_numpy())),
        index=df_main.index,
    )

    # Sanity checks
    if not (len(X) == len(y) == len(trial_id)):
        raise ValueError("X, y, and trial_id must have the same number of rows.")
    unique_trials = pd.unique(trial_id)
    if len(unique_trials) < trials:
        raise ValueError(
            f"Need at least trials unique (patient, video) trials for a {trials}-trial holdout."
        )

    rng = np.random.default_rng(random_state)
    held_out_trials = rng.choice(unique_trials, size=trials, replace=False)

    test_mask = trial_id.isin(set(held_out_trials)).to_numpy()

    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]

    print(
        "Held-out (patient, video) trials:",
        sorted(held_out_trials, key=lambda t: (t[0], t[1])),
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def omit_patient(  # PERFORMS LOSO (Leave-one--out)
    target: str = "arousal",
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)
    df_main = df_main.drop(columns=["Unnamed: 0"], errors="ignore")

    X = df_main.drop(
        columns=["patient_index", "video_index", "arousal", "valence"],
        errors="ignore",
    )
    y = df_main[target]

    patient_index = df_main["patient_index"]
    patient_index = pd.Series(patient_index).reset_index(drop=True)

    rng = np.random.default_rng()
    unique_patients = pd.unique(patient_index)
    if len(unique_patients) < 2:
        raise ValueError("Need at least 2 unique patients for a LOSO split.")
    held_out_patient = rng.choice(unique_patients)

    test_mask = (patient_index == held_out_patient).to_numpy()
    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]
    print(held_out_patient)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    estimators: int = 100,
    max_depth: int = 8,
    n_jobs: int = -1,
):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test

def train_random_forest_regressor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    estimators: int = 100,
    max_depth: int = 8,
    n_jobs: int = -1,
):
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


def train_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    neighbors: int = 1,
    weights: str = "distance",
    n_jobs: int = -1,
):
    knn = KNeighborsClassifier(
        n_neighbors=neighbors, weights=weights, n_jobs=n_jobs, p=1
    )
    knn.fit(X_train, y_train)

    return knn, X_test, y_test


def train_knn_regressor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    neighbors: int = 1,
    weights: str = "distance",
    n_jobs: int = -1,
):
    knn = KNeighborsRegressor(
        n_neighbors=neighbors, weights=weights, n_jobs=n_jobs, p=1
    )
    knn.fit(X_train, y_train)

    return knn, X_test, y_test


def train_lstm(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | pd.DataFrame | np.ndarray,
    y_test: pd.Series | pd.DataFrame | np.ndarray,
    *,
    units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    bidirectional: bool = False,
    patience: int = 8,
    verbose: int = 0,
    random_seed: int = 42,
):
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    X_train_arr = np.asarray(X_train).astype(np.float32, copy=False)
    X_test_arr = np.asarray(X_test).astype(np.float32, copy=False)
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    y_train_arr = np.array(
        [1 if str(v).lower() == "high" else 0 for v in y_train_arr], dtype=np.float32
    )
    y_test_arr = np.array(
        [1 if str(v).lower() == "high" else 0 for v in y_test_arr], dtype=np.float32
    )

    if X_train_arr.ndim == 2:
        X_train_arr = X_train_arr[:, None, :]
    if X_test_arr.ndim == 2:
        X_test_arr = X_test_arr[:, None, :]

    timesteps = X_train_arr.shape[1]
    n_features = X_train_arr.shape[2]

    inp = layers.Input(shape=(timesteps, n_features))
    lstm_block = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
    )
    x = layers.Bidirectional(lstm_block)(inp) if bidirectional else lstm_block(inp)
    x = layers.Dense(units // 2, activation="leaky_relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6
        ),
    ]

    model.fit(
        X_train_arr,
        y_train_arr,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cbs,
        shuffle=True,
    )

    return model, X_test_arr, y_test_arr


def single_user_split(target: str, k_holdouts: int, selected_user: int = None, random_state=None):
    """Splits one trial-user combination and returns only one user's data in the train set"""
    df = read_table("datasets/features_table.csv").reset_index(drop=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    rng = np.random.default_rng(random_state)
    if selected_user is None:
        users = df["patient_index"].unique()
        selected_user = int(rng.choice(users))

    videos = df["video_index"].unique()
    holdouts = rng.choice(videos, size=k_holdouts, replace=False)

    X = df.drop(
        columns=["patient_index", "video_index", "arousal", "valence"], errors="ignore"
    )
    y = df[target]

    mask = df["patient_index"] == selected_user

    trial_mask = mask & (df["video_index"].isin(holdouts))

    X_train = X.loc[mask & ~trial_mask].reset_index(drop=True)
    y_train = y.loc[mask & ~trial_mask].reset_index(drop=True)

    X_test = X.loc[trial_mask].reset_index(drop=True)
    y_test = y.loc[trial_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, arousal_train, arousal_test = single_user_split(target="arousal")
