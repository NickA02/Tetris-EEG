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
- Comprehensive feature importance analysis using 5 different methods
- Pre-filtered datasets with top N important features (configurable)
- Visualizations for feature rankings, frequency bands, and brain regions

Please read the [EEG Prediction README](EEG%20Prediction/README.md) before running experiments or submitting changes.
