import pandas as pd
from .utils import read_table
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def omit_patient_video(test_size: float = 0.1, target: str = "arousal"):
    df = read_table("datasets/features_table.csv")
    df = df.reset_index(drop=True)
    # df2 = read_table("datasets/features_table_imf.csv")
    # df2 = df.drop(columns=["patient_index","video_index"])
    # df2 = df2.reset_index(drop=True)

    # df = pd.concat([df, df2], axis=1)
    # df = df.reset_index(drop=True)

    X = df.drop(columns=["patient_index", "video_index", "arousal", "valence", "Unnamed: 0"])
    y = df[target].astype(float)

    groups = df["patient_index"].astype(str) + "_" + df["video_index"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
    knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, n_jobs=n_jobs)
    knn.fit(X_train, y_train)

    return knn, X_test, y_test


def train_svr(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
):
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, verbose=True)  # TODO: try "linear" also
    svr.fit(X_train, y_train.values.ravel())

    return svr, X_test, y_test


if __name__ == "__main__":
    X_train, X_test, arousal_train, arousal_test = omit_patient_video(target="arousal")
