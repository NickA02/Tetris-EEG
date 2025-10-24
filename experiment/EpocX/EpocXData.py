from datetime import datetime
import pandas as pd
import os


def save_eeg_data(
    filename: str, 
    user_id: int,
    session_id: int,
    object_count: int,
    time_elapsed: float,
    arousal: int,
    valence: int,
    fall_speed: float,
    difficulty_type: str,
    sensor_contact_quality: bool,
    df: pd.DataFrame
) -> int:
    if df is None or df.empty:
        return 0

    df_to_write = df.copy()
    df_to_write.insert(0, "user_id", user_id)
    df_to_write.insert(1, "session_id", session_id)
    df_to_write.insert(2, "object_count", object_count)
    df_to_write.insert(3, "time_elapsed", time_elapsed)
    df_to_write.insert(5, "arousal", arousal)
    df_to_write.insert(6, "valence", valence)
    df_to_write.insert(7, "fall_speed", fall_speed)
    df_to_write.insert(8, "difficulty_type", difficulty_type)
    df_to_write.insert(9, "sensor_contact_quality", sensor_contact_quality)

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

