# Experiment

Real-time EEG data collection utilities for in-game/interactive experiments using Emotiv EpocX. This folder handles device/session setup, streaming, and saving synchronized datasets that later feed the DREAMER-style modeling pipeline.

```
experiment/
├─ EpocX/
│  ├─ EpocXData.py      # Functions to handle incoming EEG data and predict (V/A)
│  ├─ EpocXService.py   # EpocX streaming service wrapper
│  └─ __init__.py
└─ experiment.py         # Orchestrates a full recording session (CLI entry point)
```

- `EpocXData.py` contains functions to handle data collected from EpocX, for example `predict_flow` which uses the models from `dreamer_models/predictor_model.py`
- `EpocXService.py` full functionality for streaming PSD (microV^2/Hz) from a nearby Emotiv EPOC X headset that is connected to the server device's Emotiv Launcher.
- `experiment.py` contains functions that should be called in-game (in `tetris`). Mainly, `predict_n_insert` which calls functions from `EpocXData.py` to predict valence and insert collected EEG data to the database.