import pandas as pd
from pathlib import Path

def read_table(filename: str = "datasets/features_table.csv") -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir.parent / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)
    return df