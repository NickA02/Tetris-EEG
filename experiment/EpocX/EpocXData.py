from datetime import datetime
import pandas as pd
import os

def save_eeg_data(filename: str, user_id: int, session_id: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    
    df_to_write = df.copy()
    df_to_write.insert(0, "user_id", user_id)
    df_to_write.insert(1, "session_id", session_id)

    file_exists = os.path.exists(filename)

    if file_exists:
        existing_cols = pd.read_csv(filename, nrows=0).columns.tolist()

        incoming_cols = df_to_write.columns.tolist()
        if set(existing_cols) != set(incoming_cols):
            missing = set(existing_cols) - set(incoming_cols)
            extra = set(incoming_cols) - set(existing_cols)
            raise ValueError(
                f"Column mismatch.\nMissing in incoming: {missing}\nExtra in incoming: {extra}"
            )

        df_to_write = df_to_write[existing_cols]
        df_to_write.to_csv(
            filename,
            mode="a",
            index=False,
            header=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        
    print(f"Saved EPOC X EEG data (user_id={user_id}, session_id={session_id}) to database.")
