# DREAMER Models

Python 3.10+, NumPy/Pandas, SciPy, scikit-learn, MNE (for ICA), PyTorch (RF variant), Matplotlib.

```
dreamer_model/
├─ datasets/              # Raw/processed DREAMER CSVs (local only)
├─ ML/
│  ├─ ICA.py              # EEG-only ICA helpers for artifact removal
│  ├─ model_training.py   # Train/eval utilities, splits, metrics, saves
│  ├─ rf_torch.py         # PyTorch Random Forest wrapper/impl
│  └─ utils.py            # Filtering, subsets, asymmetry features
├─ features.ipynb         # Generates feature tables
├─ predictor_model.ipynb  # Trains model for in-game prediction (imported in tetris)
├─ run_models.ipynb       # ML model walkthrough
└─ start_dreamer.ipynb    # Sets up dreamer
```

## predictor_model.py
This script trains a model that can be imported by other parts of this project, namely, in the `experiment` portion. The script runs:
```py
arousal_model, X_test, y_test = train_knn(X, None, arousal_target, None, neighbors=best_n)
valence_model, X_test, y_test = train_knn(X, None, valence_target, None, neighbors=best_n)
```
Then, `arousal_model` and `valence_model` can be imported to do in-game speed change predictions over the real-time collected EEG data.


## model_training.py
- random_train_test_split(...): Loads features_table.csv, builds X,y for target, drops metadata, and returns a standard randomized train/test split.
- omit_patient_video(...): Loads main features, groups by patient_index_video and does a GroupShuffleSplit so entire patient–video pairs are held out together; returns grouped train/test splits.
- train_random_forest(X_train, X_test, y_train, y_test, ...): Fits a RandomForestRegressor (pre-set to 300 trees, full depth) and returns the trained model with the untouched test sets.
- train_knn(X_train, X_test, y_train, y_test, ...): Fits a distance-weighted KNeighborsRegressor (default k=1, p=1) and returns the model with the test sets.
train_svr(X_train, X_test, y_train, y_test): Fits an RBF-kernel SVR (C=1.0, ε=0.1) and returns the model with the test sets.


## ML/utils.py
- read_table(filename="datasets/features_table.csv"): Resolves the repo path and loads a CSV into a pandas.DataFrame, raising FileNotFoundError if missing.
- generate_all_subsets(columns, sensor_families=None, feature_tokens=None, freq_tokens=None, min_size=6): Parses columns into (sensor family, feature token, frequency band) and returns unique feature subsets formed by all combinations of present families/features/bands, filtered to at least min_size.
- HOMOLOGOUS_PAIRS: Left–right channel pairs for Emotiv/DREAMER (e.g., (AF3, AF4), (F3, F4)) used to compute hemispheric asymmetries.
- AS_BANDS: Allowed band names for asymmetry computation: {delta, theta, alpha, beta, gamma}.
- compute_asymmetry_from_psd(psd, pairs=HOMOLOGOUS_PAIRS, eps=1e-12, add_log=True, prefix_da="da", prefix_ra="ra"): From PSD features like CH_band, computes directional asymmetry da (log- or linear difference) and rational asymmetry ra = (PR−PL)/(PR+PL+eps) for each pair×band, returning a new DataFrame of asymmetry columns.

