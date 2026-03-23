# Price Prediction Regression Pipeline

End-to-end machine learning pipeline for price prediction using tabular data, with data cleaning, feature preprocessing, model comparison, hyperparameter tuning, and diagnostic evaluation.

## Overview

This project builds a complete machine learning workflow for predicting prices from structured tabular data. It starts with raw data, applies cleaning and preprocessing, compares multiple regression models, tunes the strongest candidates, and evaluates the final model using multiple regression metrics and diagnostic plots.

The goal of this project is to demonstrate a portfolio-ready regression pipeline that goes beyond a basic notebook by emphasizing reproducibility, model selection, interpretability, and practical machine learning workflow design.

## Features

- Automated cleaning for high-missing, high-zero, and constant columns
- Separate preprocessing for numeric and categorical variables
- Leakage-safe train/test workflow using scikit-learn pipelines
- Model comparison across multiple regression algorithms
- Hyperparameter tuning with cross-validation
- Final evaluation using MAE, RMSE, R², Median Absolute Error, and MAPE
- Diagnostic plots for actual vs. predicted values and residual behavior
- Permutation-based feature importance
- Saved trained pipeline for reuse on new price prediction data

## Project Structure

```text
price-prediction-regression-pipeline/
├── README.md
├── requirements.txt
├── regression_pipeline.py
├── mlxproject.ipynb
└── regression_project_outputs/
    ├── eda/
    ├── metrics.json
    ├── model_comparison_cv.csv
    ├── tuned_model_results.csv
    ├── test_predictions.csv
    ├── permutation_feature_importance.csv
    ├── best_regression_pipeline.joblib
    └── diagnostic plots
```

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib

## Models Compared

The pipeline compares several regression models to identify the strongest approach for price prediction:

- Dummy Regressor
- Elastic Net
- Random Forest Regressor
- Extra Trees Regressor
- HistGradientBoosting Regressor

The top-performing models are then tuned using randomized cross-validated search.

## Workflow

### 1. Data Loading
The script supports structured datasets such as CSV, Excel, and Parquet files.

### 2. Data Cleaning
The pipeline automatically removes:
- columns with too many missing values
- numeric columns dominated by zeros
- constant columns with no useful variation

### 3. Preprocessing
Numeric and categorical variables are handled separately:
- numeric features are imputed and scaled
- categorical features are imputed and one-hot encoded

### 4. Model Selection
Multiple regression models are evaluated using cross-validation on the training set.

### 5. Hyperparameter Tuning
The strongest candidate models are tuned with randomized search to improve predictive performance.

### 6. Final Evaluation
The selected model is evaluated on a held-out test set using multiple regression metrics.

### 7. Diagnostics and Interpretability
The project generates:
- actual vs. predicted plots
- residual plots
- residual distributions
- permutation feature importance rankings

### 8. Model Persistence
The final trained pipeline is saved as a `.joblib` artifact for reuse on future data.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` file for this project is:

```txt
pandas
numpy
matplotlib
scikit-learn
joblib
openpyxl
pyarrow
```

## How to Run

Open `regression_pipeline.py` and update these two lines:

```python
DATA_PATH = "test.csv"
TARGET_COL = "PRICE_COLUMN"
```

Then run:

```bash
python regression_pipeline.py
```

The script will create an output folder containing evaluation reports, plots, and the saved trained model.

## Output Files

After running the pipeline, the project saves several artifacts:

- `model_comparison_cv.csv` — cross-validated model comparison
- `tuned_model_results.csv` — best tuned hyperparameters and scores
- `metrics.json` — final train/test metrics
- `test_predictions.csv` — predictions, residuals, and errors
- `permutation_feature_importance.csv` — feature importance rankings
- `best_regression_pipeline.joblib` — saved trained pipeline
- diagnostic plots and EDA reports

## Example Use Cases

This pipeline can be adapted for a wide range of price prediction problems, including:

- house price prediction
- vehicle price estimation
- product price modeling
- resale value prediction
- marketplace analytics
- business forecasting and valuation support

## Notebook Version

The repository also includes `mlxproject.ipynb`, which shows the project in notebook form. This version is useful for exploration, intermediate analysis, and understanding the progression from data cleaning and exploratory work to a full machine learning pipeline.

## Why This Project Matters

Many machine learning projects stop at fitting a single model in a notebook. This project goes further by building a more realistic workflow that includes preprocessing, validation, model comparison, tuning, diagnostics, and reproducibility. It demonstrates both machine learning and applied data analysis skills in a way that is closer to real-world project work.

## Future Improvements

Potential next steps include:

- adding SHAP-based model interpretability
- incorporating feature selection methods
- building a simple app for predictions
- adding experiment tracking
- packaging the pipeline into a reusable module
- deploying the trained model through an API

## Author

**Swarit Pakala**  
Mathematics and Computer Science, University of Wisconsin–Madison
