# Experiment

Real-time EEG data collection utilities for in-game/interactive experiments using Emotiv EpocX. This folder handles device/session setup, streaming, and saving synchronized datasets that later feed the DREAMER-style modeling pipeline.

```
experiment/
├─ EpocX/
│  ├─ EpocXData.py      # 
│  ├─ EpocXService.py   # EpocX streaming service wrapper (connect/start/stop/poll)
│  └─ __init__.py
└─ experiment.py         # Orchestrates a full recording session (CLI entry point)
```

- `EpocXData` contains functions to handle data collected from EpocX, for example