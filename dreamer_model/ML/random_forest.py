
import pandas as pd
from .utils import read_table
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np



def train_random_forest(test_size: float = 0.1):
    df = read_table()
    df = df.reset_index(drop=True)
    


    X = df.drop(columns=["patient_index", "video_index", "arousal", "valence"])
    y = df["arousal"].astype(float)
    valence = df["valence"].values


    groups = df["video_index"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


if __name__ == "__main__":
    rf, X_test, y_test = train_random_forest()

    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r, pval = pearsonr(y_test, y_pred)
    print(f"PCC: {r:.3f}  (p={pval:.3g})")

    print("Random Forest Regression Performance:")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # (Optional) quick feature importance peek
    importances = getattr(rf, "feature_importances_", None)
    if importances is not None:
        top = 10
        order = np.argsort(importances)[::-1][:top]
        print("\nTop feature importances:")
        for i in order:
            print(f"{X_test.columns[i]}: {importances[i]:.4f}")
