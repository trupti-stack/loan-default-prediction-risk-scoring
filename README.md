# Loan Default Prediction & Risk Scoring

## Overview

This project develops an end-to-end machine learning pipeline for predicting loan default risk and supporting risk-based lending decisions.

The workflow covers data cleaning, exploratory data analysis, data leakage prevention, feature engineering, feature selection, preprocessing, model benchmarking, hyperparameter optimization, cross-validation, threshold tuning, error analysis, applicant risk scoring, and expected financial impact estimation.

The target variable is `loan_status`, where completed loans are classified as:

- `Fully Paid` → 0
- `Charged Off` → 1

## Problem Statement

Loan default prediction is a binary classification problem in which the objective is to identify borrowers who are at higher risk of default.

The project focuses on improving the detection of potential defaults while maintaining an acceptable level of precision. Since missing risky borrowers can be costly, model selection and decision-threshold tuning are evaluated with a recall-focused objective.

## Dataset

The project uses a Lending Club loan dataset loaded from `loan.csv`.

The data is first inspected for:

- Dataset shape and column information
- Data types
- Missing values
- Duplicate records
- Descriptive statistics
- Categorical values
- Outliers
- Feature correlations
## Methodology

### 1. Data Cleaning

The preprocessing workflow includes:

- Removing duplicate records
- Identifying missing values
- Filling numerical missing values using the median
- Filling categorical missing values using the mode
- Inspecting numerical and categorical distributions
- Detecting loan amount outliers using the IQR method

### 2. Data Leakage Prevention

Features containing information that could become available after loan origination or during repayment were identified as potential leakage variables and removed.

Examples include:

- `funded_amnt`
- `funded_amnt_inv`
- `out_prncp`
- `out_prncp_inv`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_prncp`
- `total_rec_int`
- `recoveries`
- `last_pymnt_d`
- `last_pymnt_amnt`
- `next_pymnt_d`

A leakage-reduced dataset is saved as `loan_no_leakage.csv`.

### 3. Feature Engineering

Several derived features are created to capture borrower and loan characteristics, including:

- Loan-to-income ratio
- Installment-to-income ratio
- Credit history years
- Income per account
- Debt per account
- Open account ratio
- High-income indicator
- High-DTI indicator
- Long-credit-history indicator

Date variables are converted to datetime values and used to derive credit history length before being removed.

### 4. Feature Preprocessing

The preprocessing workflow includes:

- Converting percentage and term fields into numeric values
- One-hot encoding categorical variables
- Converting the target variable into binary form
- Removing unnecessary columns
- Splitting the data into training, validation, and testing sets
- Median imputation for numeric missing values
- Standard scaling using training data
- Removing highly correlated features with absolute correlation greater than 0.90

The data is split into:

- 70% Training
- 15% Validation
- 15% Testing

Stratification is used to preserve the class distribution.
## Machine Learning Models

Five classification algorithms are benchmarked:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Extra Trees

The baseline models are compared using validation-set performance.

The best baseline model is selected using recall, followed by precision and ROC-AUC.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

The project also performs cross-validation using `StratifiedKFold` with 3 folds to evaluate model robustness across multiple training-validation splits.

## Hyperparameter Optimization

The selected baseline model is optimized using `GridSearchCV`.

The tuning process uses:

- Stratified 3-fold cross-validation
- Recall as the optimization metric
- Model-specific parameter grids
- Parallel processing
- Refitting using the best parameter combination

The best parameters and cross-validation recall are recorded after tuning.

## Threshold Optimization

Instead of relying only on the default classification threshold of 0.50, the project evaluates different probability thresholds.

The decision threshold is selected to:

- Maintain a minimum precision of 0.60
- Maximize recall among thresholds satisfying that precision requirement

This allows the classification decision to reflect the business objective of identifying more potential loan defaults while maintaining acceptable precision.
## Risk Scoring

The final model produces a probability of loan default for each applicant.

Applicants are assigned to three risk-based decision bands:

| Default Probability | Decision |
|---|---|
| `< 0.40` | Approve |
| `0.40 – < 0.70` | Manual Review |
| `≥ 0.70` | Reject |

The project also performs error analysis by examining:

- False negatives: risky loans predicted as safe
- False positives: safe loans predicted as risky

## Business Insights

The predicted default probabilities are converted into actionable lending decisions using risk bands.

The project also estimates potential financial impact using:

- Loss Given Default (LGD) = 0.35
- Profit Margin = 0.08

The analysis calculates:

- Avoided default loss
- Missed good-loan profit
- Estimated net savings

This connects model predictions with a lending decision strategy rather than evaluating the model only on statistical metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/loan-default-prediction-risk-scoring.git
cd loan-default-prediction-risk-scoring
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the required `loan.csv` dataset in the project directory.

### 4. Run the notebook

Open:

`Loan_Default_Prediction.ipynb`

The notebook can be run using Google Colab or a local Jupyter environment.

## Project Notebook

## Project Notebook

[Open the Project in Google Colab](https://colab.research.google.com/drive/1Zu1xCuoSGGxkn6yWJ2wK0PgM-8VrC-Ko)
