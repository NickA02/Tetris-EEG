import pandas as pd
from .utils import read_table
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def random_train_test_split(
    test_size: float = 0.1,
    target: str = "arousal",
    shuffle_random_state: int | None = None,
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)

    df = df_main.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    # df = pd.concat([df_main, df_imf_no_meta], axis=1).reset_index(drop=True)

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


def omit_patient_video(
    test_size: float = 0.1,
    target: str = "arousal",
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)
    # df_imf = read_table("datasets/features_table_imf.csv").reset_index(drop=True)
    # df_imf_no_meta = df_imf.drop(
    #     columns=["patient_index", "video_index", "arousal", "valence", "Unnamed: 0"],
    #     errors="ignore",
    # )
    df_main = df_main.drop(columns=["Unnamed: 0"], errors="ignore")

    # df = pd.concat([df_main, df_imf_no_meta], axis=1).reset_index(drop=True)

    X = df_main.drop(
        columns=["patient_index", "video_index", "arousal", "valence"],
        errors="ignore",
    )

    y = df_main[target].astype(float)

    groups = df_main["patient_index"].astype(str) + "_" + df_main["video_index"].astype(str)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        # random_state=42,
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

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
    knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, n_jobs=n_jobs, p=1)
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
