# DEAP Emotion Recognition

This repository contains code to reproduce emotion recognition experiments on the **DEAP** dataset using **bandpower features**. It also includes code to run various models to compare results across multiple settings. 


## Deep Dataset Download

We use DEAP dataset from Kaggle. This by default will download it to the ```.cache``` folder. So the code modes it to the current directory. 

To download the dataset, run  - ```python3 deap_dataset.py```

---

## Experiments

### DEAP Baseline Classifier

This script implements **baseline emotion classification** experiments on the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/), which contains EEG and peripheral physiological recordings for emotion recognition.

The script:
- Extracts **EEG features** (log-bandpower in theta, alpha, beta, gamma bands)  
- Extracts **peripheral features** (GSR, Respiration, BVP, Skin Temperature statistics)  
- Binarizes DEAP labels (**valence, arousal, dominance, liking**) into *High* / *Low*  
- Trains classifiers using **SVMs with linear kernels**  
- Evaluates performance in two settings:
  - **Subject-dependent:** 10-fold CV within each subject
  - **Cross-subject (LOSO):** Leave-One-Subject-Out with EEG-only, peripherals-only, and EEG+peripherals fusion

Outputs include **Accuracy, Balanced Accuracy, and Macro F1-Score**.

Run the script - `python deap_baseline.py`


### DEAP Baseline Python Script

This notebook is complementary to `deap_baseline.py`, but structured for interactive experimentation.

This Jupyter notebook conducts **baseline experiments** on the **DEAP dataset** for emotion recognition.  
It focuses on:
- **Feature extraction** from EEG and peripheral signals
- **Classical ML models** (Linear SVM, calibrated classifiers)
- **Cross-validation evaluation** (subject-dependent and LOSO)

---
### MIL (Multiple-Instance Learning) for DEAP 

This Jupyter notebook implements **Multiple-Instance Learning (MIL)** with attention for emotion recognition on **EEG windowed features** from the **DEAP** dataset. It treats each **trial** as a *bag* of short-time **windows** (instances), learns attention weights over windows, and predicts **valence/arousal** (supports multi-output).

Run the script - It is a notebook. Run on Jupyter. 

---

### MLP for Emotion Classification (DEAP)

This Jupyter notebook implements **Multi-Layer Perceptrons (MLPs)** for classifying emotional states using the **DEAP dataset**.

It explores:
- **Single-output models**: predict either binary *valence* or *arousal* separately.
- **Multi-output models**: jointly predict binary *valence* and *arousal*.
- **Categorical outputs (1–9 scale)**: instead of binary classification, emotions are classified into one of 9 categories.

---

### ANN, SVM, RF, and KNN for Emotion Classification (DEAP)

This Jupyter notebook compares **shallow and traditional machine learning models** for classifying emotions using the **DEAP dataset**.  
It implements and evaluates:
- **ANN (Artificial Neural Network)** — a simple PyTorch feedforward model
- **SVM (Support Vector Machine)**
- **RF (Random Forest)**
- **KNN (K-Nearest Neighbors)**

It also explores **feature-level fusion** across EEG channels and reports **cross-validation results**.

---

### LSTM + AutoKeras (NOT RUN FULLY YET. LOSS IS NOT DIVERGING. NEEDS MODIFICATION)

This Jupyter notebook handles **data preprocessing** and **model prototyping** for the **DEAP dataset**.  
It includes:
- Preparing EEG/label data from DEAP `.dat` files
- Splitting into train/test sets
- Building baseline models using LSTM and Autokeras
- **Models**
  - **LSTM**:
    - Implemented with TensorFlow/Keras (`layers.LSTM`, `Dense`, `Dropout`).
    - Configurable epochs, batch size, optimizer, and callbacks.
  - **AutoKeras**:
    - Automatically searches for a suitable neural architecture given the data.
    - Reduces manual hyperparameter tuning.

---

## Directory Structure 

```
EEG Prediction/
│
├── datasets/DEAP/deap-dataset/         # DEAP dataset files (not included in repo, must be downloaded)
│   ├── audio_stimuli_MIDI/             # Audio stimuli used in DEAP
│   ├── audio_stimuli_MIDI_tempo24/     # Alternate MIDI stimuli set
│   ├── data_preprocessed_python/       # Preprocessed EEG signals (s01.dat ... s32.dat)
│   ├── extracted_features/             # Feature-level data (CSV per channel)
│   ├── Metadata/                       # Metadata files
│   ├── metadata_xls/                   # Participant ratings in Excel
│   └── EDA_DEAP.ipynb                  # Exploratory data analysis notebook
│
├── notebooks/                          # Jupyter notebooks for experiments
│   ├── ANN_SVM_RF_KNN.ipynb            # Classical ML models + ANN
│   ├── deap_baseline.ipynb             # Baseline experiments with SVMs
│   ├── LSTM_Autokeras.ipynb            # Preprocessing + LSTM + AutoKeras models
│   ├── MIL.ipynb                       # Multiple-Instance Learning with attention
│   └── MLP.ipynb                       # Multi-Layer Perceptrons (single & multi-output)
│
├── scripts/                            # Standalone Python scripts
│   └── deap_baseline.py                # Baseline feature extraction + classification
│
├── .gitignore                          # Ignored files (datasets, checkpoints, etc.)
├── deap_dataset.py                     # Utility for dataset loading/preprocessing
├── README.md                           # Project overview & instructions
└── requirements.txt                    # Dependencies
```

## Contibute

### 1. Setup Environment
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/EEG-Prediction.git
cd EEG-Prediction
pip install -r requirements.txt
```

---

### 2. Adding New Experiments
- **New notebooks** → put inside `notebooks/`  
  Example: `notebooks/Transformer_EEG.ipynb`
- **Reusable scripts** → put inside `scripts/` if it’s meant for multiple notebooks  
  Example: `scripts/feature_extraction.py`
- **Dataset paths** → always use relative paths (e.g., `./datasets/DEAP/...`) so code runs on any machine.

---

### 3. Naming Conventions
- Python files: `snake_case.py`
- Notebooks: `ModelName_Description.ipynb` (e.g., `CNN_EmotionNet.ipynb`)
- Functions/classes: follow PEP8 standards.

---

### 4. Updating Requirements
If you add a new dependency:
1. Install it locally
   ```bash
   pip install <package>
   ```
2. Pin the version and update requirements
   ```bash
   pip freeze | grep <package> >> requirements.txt
   ```

---


