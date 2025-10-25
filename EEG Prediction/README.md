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

### Feature Importance Analysis

This notebook performs **comprehensive feature importance analysis** to identify which EEG features (channel-band combinations) have the most impact on predicting valence and arousal.

**Methods Used:**
1. **Correlation Analysis** — Pearson & Spearman correlation with targets
2. **Statistical Tests** — ANOVA F-test and Mutual Information
3. **Random Forest** — Tree-based feature importance scores
4. **Permutation Importance** — Model-agnostic importance by shuffling features
5. **Linear SVM Coefficients** — Weight analysis from linear models

**Outputs:**
- Aggregated feature rankings (averaging across all 5 methods)
- Frequency band analysis (alpha/beta/gamma importance)
- Brain region analysis (which channels contribute most)
- Visualizations (bar charts for top features)
- CSV files: `feature_importance_valence.csv` and `feature_importance_arousal.csv`

**Configuration:**
Adjust the number of features to analyze by modifying these parameters at the top of the notebook:
```python
TOP_N_FEATURES = 30      # Number of features shown in tables
TOP_N_VISUALIZE = 20     # Number of features in plots
TOP_N_AGGREGATE = 50     # Features in final aggregated ranking
```

Run the notebook: `notebooks/Feature_Importance_Analysis.ipynb`

---

### Create Filtered Dataset

This notebook generates **filtered datasets** containing only the most important features identified by the Feature Importance Analysis.

**What it Creates:**
1. **Valence-Optimized Dataset** — Top N features for valence prediction
2. **Arousal-Optimized Dataset** — Top N features for arousal prediction
3. **Combined Dataset** — Top N features for both valence and arousal

**Output Formats:**
- **NumPy arrays** (.npy) for fast loading
- **CSV files** with feature names and labels
- **Feature lists** (.txt) documenting which features were selected
- **README.md** with complete metadata

**Configuration:**
Adjust the number of features to include:
```python
TOP_N_VALENCE = 30   # Features for valence-optimized dataset
TOP_N_AROUSAL = 30   # Features for arousal-optimized dataset
TOP_N_COMBINED = 40  # Features for combined dataset
```

**Usage Example:**
```python
import numpy as np

# Load filtered dataset
X = np.load('datasets/DEAP/filtered_features/X_valence_top30.npy')
y = np.load('datasets/DEAP/filtered_features/y_valence_binary.npy')

# Train model with reduced feature set (30 instead of 96)
```

**Benefits:**
- Faster training time
- Reduced risk of overfitting
- Better model interpretability
- Focus on most predictive features

Run the notebook: `notebooks/Create_Filtered_Dataset.ipynb`

---

## Directory Structure

```
EEG Prediction/
│
├── datasets/DEAP/
│   ├── deap-dataset/                   # DEAP dataset files (not included in repo, must be downloaded)
│   │   ├── audio_stimuli_MIDI/         # Audio stimuli used in DEAP
│   │   ├── audio_stimuli_MIDI_tempo24/ # Alternate MIDI stimuli set
│   │   ├── data_preprocessed_python/   # Preprocessed EEG signals (s01.dat ... s32.dat)
│   │   ├── extracted_features/         # Feature-level data (CSV per channel)
│   │   ├── Metadata/                   # Metadata files
│   │   ├── metadata_xls/               # Participant ratings in Excel
│   │   └── EDA_DEAP.ipynb              # Exploratory data analysis notebook
│   │
│   └── filtered_features/              # Filtered datasets with top important features
│       ├── X_valence_top30.npy         # Valence-optimized features (NumPy)
│       ├── X_arousal_top30.npy         # Arousal-optimized features (NumPy)
│       ├── X_combined_top40.npy        # Combined features (NumPy)
│       ├── y_valence_binary.npy        # Binary valence labels
│       ├── y_arousal_binary.npy        # Binary arousal labels
│       ├── y_valence_continuous.npy    # Continuous valence ratings
│       ├── y_arousal_continuous.npy    # Continuous arousal ratings
│       ├── dataset_valence_top30.csv   # Valence dataset with labels (CSV)
│       ├── dataset_arousal_top30.csv   # Arousal dataset with labels (CSV)
│       ├── dataset_combined_top40.csv  # Combined dataset with labels (CSV)
│       ├── features_valence_top30.txt  # List of selected valence features
│       ├── features_arousal_top30.txt  # List of selected arousal features
│       ├── features_combined_top40.txt # List of selected combined features
│       └── README.md                   # Metadata about filtered datasets
│
├── notebooks/                          # Jupyter notebooks for experiments
│   ├── ANN_SVM_RF_KNN.ipynb            # Classical ML models + ANN
│   ├── deap_baseline.ipynb             # Baseline experiments with SVMs
│   ├── LSTM_Autokeras.ipynb            # Preprocessing + LSTM + AutoKeras models
│   ├── MIL.ipynb                       # Multiple-Instance Learning with attention
│   ├── MLP.ipynb                       # Multi-Layer Perceptrons (single & multi-output)
│   ├── Feature_Importance_Analysis.ipynb  # Comprehensive feature importance analysis
│   ├── Create_Filtered_Dataset.ipynb   # Generate datasets with top features
│   ├── feature_importance_valence.csv  # Feature rankings for valence (output)
│   ├── feature_importance_arousal.csv  # Feature rankings for arousal (output)
│   ├── correlation_analysis.png        # Visualization: correlation analysis
│   ├── random_forest_importance.png    # Visualization: RF importance
│   ├── top_features_aggregated.png     # Visualization: aggregated rankings
│   ├── band_importance.png             # Visualization: frequency band importance
│   └── channel_importance.png          # Visualization: channel importance
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


