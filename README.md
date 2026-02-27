# Car Insurance Claim Prediction

> A machine learning project that predicts whether a customer will make a car insurance claim — built with a full data preprocessing pipeline, comparative model evaluation, and a custom business revenue optimization framework.

**Author:** Ridwan Adeniyi

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Model Results](#model-results)
- [Revenue Optimization](#revenue-optimization)
- [Final Leaderboard Performance](#final-leaderboard-performance)
- [License](#license)

---

## Overview

This project tackles a binary classification problem: given a customer's profile and vehicle details, predict whether they will make an insurance claim (`is_bound`). The workflow spans the full data science lifecycle — from raw, messy Excel data to a production-ready model with a business-optimized prediction threshold.

Key highlights include a reusable `preprocess_data()` function designed to prevent data leakage, frequency encoding for high-cardinality features, and a custom revenue curve to find the threshold that maximizes business profit rather than just accuracy.

---

## Features

- Comprehensive data cleaning pipeline (outlier handling, type normalization, imputation)
- Vehicle make/model typo correction and brand normalization
- Joint imputation of `annual_km` and `commute_distance` using a ratio-based factor
- Frequency encoding for high-cardinality categorical features
- One-hot encoding + median imputation in a scikit-learn `Pipeline`
- Logistic Regression vs. Random Forest comparison via ROC-AUC
- Custom `advertising_revenue()` function for business-driven threshold selection
- Final model refit on full training data and test set prediction export

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and wrangling |
| `numpy` | Numerical operations and array handling |
| `matplotlib` / `seaborn` | Visualization (ROC curves, revenue plots) |
| `scikit-learn` | Preprocessing pipelines, model training, evaluation |

---

## Getting Started

### Prerequisites

Python 3.8+ is required. Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Running the Project

**Option 1 — Google Colab (recommended)**

Open `Car_Insurance_Project.ipynb` in Colab, upload your `Project2_Training.xlsx` and `Project2_Test.xlsx` files to `/content/`, then run all cells.

**Option 2 — Local Jupyter**

```bash
git clone https://github.com/your-username/car-insurance-prediction.git
cd car-insurance-prediction
jupyter notebook Car_Insurance_Project.ipynb
```

> **Note:** Update the file paths in the data loading section if running locally (change `/content/` to your local data directory).

---

## Project Structure

```
car-insurance-prediction/
│
├── Car_Insurance_Project.ipynb   # Main notebook
├── car_insurance_project.py      # Python script version
├── README.md                     # Project documentation
├── .gitignore                    # Files excluded from version control
├── LICENSE                       # MIT License
│
└── data/                         # (not tracked — see .gitignore)
    ├── Project2_Training.xlsx
    └── Project2_Test.xlsx
```

---

## Pipeline Walkthrough

### 1. Data Cleaning (`preprocess_data()`)

A single reusable function handles all preprocessing for both training and test sets, avoiding data leakage:

- **Column standardization** — lowercase, strip whitespace, replace spaces with underscores
- **Junk vehicle make removal** — filters out non-vehicle gibberish entries (e.g., VINs, nonsense strings)
- **Outlier handling** — nullifies impossible birth years, vehicle years outside 1980–2025, vehicle values under $500, annual mileage over 200,000 km, and commute distances over 200 km
- **Joint imputation** — `annual_km` and `commute_distance` are imputed from each other using a ratio derived from the training data
- **Categorical imputation** — marking/tracking systems filled with `"None"`, occupation filled with `"Not Known"`
- **Vehicle normalization** — extracts vehicle type (CAR vs VAN), corrects model-as-make errors (e.g., `CAMRY → TOYOTA`), and normalizes brand name spelling variants
- **Date encoding** — converts `quotedate` to numeric days since the earliest date in the dataset
- **Column drops** — removes `years_as_principal_driver` and `vehicle_ownership` due to high missingness
- **Final imputation** — numeric columns use median; remaining categoricals use `"Unknown"`

### 2. Feature Engineering

- **Frequency encoding** applied to high-cardinality columns (e.g., vehicle make, model, occupation) — maps each category to its frequency in the training set
- **One-hot encoding** applied to low-cardinality categoricals via scikit-learn `Pipeline`
- **Standard scaling** applied to numeric features in the Logistic Regression pipeline

### 3. Model Training & Evaluation

Two classifiers are trained and compared:

- **Logistic Regression** — L1-regularized, class-balanced, max 1000 iterations
- **Random Forest** — 900 estimators, `min_samples_leaf=5`, class-balanced

Both are evaluated on a held-out validation set using ROC-AUC and visual ROC curve comparison.

---

## Model Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.XX |
| **Random Forest** | **~0.XX** |

Random Forest outperformed Logistic Regression and was selected for final predictions.

---

## Revenue Optimization

Rather than defaulting to a 0.5 classification threshold, this project optimizes for **business revenue** using the formula:

```
Revenue = 5.5 × True Positives − 1.0 × Predicted Positives
```

Each correctly identified claimant earns **$5.50**, while each targeted customer (correct or not) costs **$1.00**. The `compute_revenue_curve()` function sweeps 201 thresholds from 0 to 1, and the threshold that maximizes validation revenue is selected for final test predictions.

---

## Final Leaderboard Performance

| Accuracy | False Negative Rate | False Positive Rate | Advertising Revenue |
|---|---|---|---|
| 52 | 27 | 52 | 17 |

---

## License

This project is licensed under the [MIT License](LICENSE).
