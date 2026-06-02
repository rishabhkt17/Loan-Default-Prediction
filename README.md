# Loan Default Prediction

This project trains a machine learning model to estimate the risk of loan default using LendingClub loan data. It includes a training pipeline, basic EDA plots, model evaluation, and a small prediction agent that can be used for single-loan predictions.

## Files

- `loan_default_pipeline.py`  
  Main training script. It loads the dataset, creates the target variable, cleans the data, performs feature engineering, trains models, evaluates them, and saves the best pipeline.

- `loan_agent.py`  
  Loads the saved pipeline and provides a simple prediction interface for new loan applications.

- `plots/`  
  Stores EDA charts generated during training.

- `loan_default_pipeline.pkl`  
  Saved sklearn pipeline containing preprocessing and the trained model.

- `requirements.txt`  
  Python dependencies required to run the project.

## Dataset

The project expects the LendingClub accepted loans dataset:

```text
accepted_2007_to_2018Q4.csv