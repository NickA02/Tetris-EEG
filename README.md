# Tetris-EEG (A WIP Name)
## Initial Setup
Current tetris implementation only tested with Python 3.13.7

Once you have set up your python environment, run
```zsh
pip install -r requirements.txt
```

## Playing the game/Data Collection

To play the game, set the current working directory to Tetris-EEG/tetris, then run main.py

```zsh
cd tetris
python main.py
```

### Controls

| Button | Action | Description |
|---|---|---|
| Left Arrow | Move Left | Moves the current piece to the left |
| Right Arrow | Move Right | Moves the current piece to the right |
| Up Arrow | Rotate Piece | Rotates the current piece |
| Down Arrow | Soft Drop | Increases fall speed of the current piece |
| Space | Hard Drop | Instantly lands and locks current piece |
| C | Hold/Swap | Stashes current piece in the Hold cell, swaps with piece in hold cell if available |


## Real-Time EEG → Tetris Pipeline

The project now ships with an extensible affect pipeline that streams valence/arousal scores into the game and adjusts multiple difficulty dimensions (fall speed, garbage rows, piece distribution, hold availability, preview depth). The overlay (toggle with **O**) displays current affect readings, selected difficulty modes, and the player’s score for quick verification.

### 1. Start the EEG inference service

Run the service from the repository root (dummy mode is helpful while iterating):

```bash
python -m eeg_service.main --mode dummy --host 127.0.0.1 --port 5555
```

Key options:

- `--mode playback --playback-file recordings/session.jsonl` – replay recorded affect data (newline-delimited JSON).
- `--mode live` – hook into `LiveEEGSource` once it is wired to your EEG device API.
- `--model <name>` – swap model implementations (see `eeg_service/model_registry.py`).
- Logs are saved to `eeg_service/logs/affect_*.jsonl`.

> Quick start: generate a DEAP-based sample playback file  
> ```bash
> python scripts/generate_sample_playback.py --output recordings/sample_deap_session.jsonl
> python -m eeg_service.main --mode playback --playback-file recordings/sample_deap_session.jsonl
> ```
> When using raw DEAP ratings (1–9 scale), set `AFFECT_VALUE_MAX = 9.0` in `tetris/settings.py` so the difficulty adapter normalizes correctly.

### 2. Launch the Tetris client

```bash
cd tetris
python main.py
```

The client connects to the EEG service over TCP (`settings.AFFECT_HOST`, `settings.AFFECT_PORT`). If no service is available the game gracefully falls back to its baseline difficulty curve.

### 3. Difficulty dimensions & logging

- **Fall speed** is enabled by default; disable via `ENABLE_FALL_SPEED_ADJUST` in `tetris/settings.py`.
- **Garbage injection**, **piece bias**, **hold control**, and **preview depth** have independent toggles in the same file.
- Session telemetry is written to `tetris/logs/game_session_*.jsonl`, capturing affect values, difficulty state, score, and lines cleared for later analysis.

### 4. Offline playback

1. Record affect predictions as JSON Lines (`timestamp`, `valence`, `arousal`, …).
2. Replay them with:
   ```bash
   python -m eeg_service.main --mode playback --playback-file recordings/session.jsonl --playback-speed 1.0
   ```
3. Start the Tetris client to reproduce the run while capturing a fresh score log.

## EEG Prediction Module

This repository also contains a dedicated subproject for **EEG-based emotion recognition** in the directory `EEG Prediction`.

Inside, you'll find:
- **notebooks/** → experiments with classical ML and deep learning models (MLP, MIL, LSTM, ANN/SVM/RF/KNN)
  - **Feature Importance Analysis** → comprehensive analysis to identify which EEG features most impact valence/arousal prediction
  - **Filtered Dataset Creation** → generate optimized datasets with only the most important features
- **scripts/** → reusable Python scripts for baselines and dataset utilities
- **datasets/** → expected DEAP dataset structure (not included, must be downloaded separately)
  - **filtered_features/** → reduced feature sets for faster training and better interpretability
- **requirements.txt** → dependencies for running notebooks and scripts

### Key Features
- Multiple ML/DL approaches for emotion recognition (valence & arousal)
- Support for both **DEAP** and **DREAMER** datasets
- Comprehensive feature importance analysis using 5 different methods
- Pre-filtered datasets with top N important features (configurable)
- Visualizations for feature rankings, frequency bands, and brain regions
- Automatic merging of DREAMER feature files (features_table + IMF features)

Please read the [EEG Prediction README](EEG%20Prediction/README.md) before running experiments or submitting changes.
