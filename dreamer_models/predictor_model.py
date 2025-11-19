import numpy as np

from .ML.model_training import train_lstm, build_lstm_sequences
from .ML.utils import filter_features

feature_cols = filter_features(
        X_train_df.columns,
        remove_bands=["gamma", "delta"],
    )
# Make sure we only keep columns actually present
feature_cols = [c for c in feature_cols if c in X_train_df.columns]

# 2) Build sequence-level data
X_train_seq, y_train_seq = build_lstm_sequences(
    X_train_df,
    feature_cols,
    target_col=target_col,
    thresh=thresh,
    # fixed_T=15,  # uncomment if you want fixed sequence length
)
X_test_seq, y_test_seq = build_lstm_sequences(
    X_test_df,
    feature_cols,
    target_col=target_col,
    thresh=thresh,
    # fixed_T=15,
)

# 3) Sanity check class balance
print("y_train counts:", np.bincount(y_train_seq.astype(int)))
print("y_test counts:", np.bincount(y_test_seq.astype(int)))

# 4) Train LSTM
arousal_model, X_test_eval, y_test_eval = train_lstm(
    X_train_seq,
    X_test_seq,
    y_train_seq,
    y_test_seq,
    lr=lr,
    epochs=epochs,
    units=units,
    batch_size=batch_size,
    dropout=dropout,
    recurrent_dropout=recurrent_dropout,
    bidirectional=bidirectional,
)
