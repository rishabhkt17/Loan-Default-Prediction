import sys
import joblib
import pandas as pd
import numpy as np


class LoanDefaultAgent:
    def __init__(self, pipeline_path="loan_default_pipeline.pkl"):
        try:
            self.pipeline = joblib.load(pipeline_path)
            self.required_columns = self.get_required_columns()
            print("Agent initialized successfully.")
            print("Required model columns:")
            print(self.required_columns)
        except Exception as e:
            print(f"Error initializing agent: {e}")
            sys.exit(1)

    def get_required_columns(self):
        preprocessor = self.pipeline.named_steps["preprocessor"]

        required_columns = []

        for name, transformer, columns in preprocessor.transformers_:
            if name != "remainder":
                required_columns.extend(list(columns))

        return required_columns

    def clean_percent_value(self, value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, str):
            value = value.replace("%", "").strip()

        try:
            return float(value)
        except Exception:
            return np.nan

    def preprocess_input(self, input_data):
        df = pd.DataFrame([input_data])

        if "int_rate" in df.columns:
            df["int_rate"] = df["int_rate"].apply(self.clean_percent_value)

        if "revol_util" in df.columns:
            df["revol_util"] = df["revol_util"].apply(self.clean_percent_value)

        if "term" in df.columns:
            df["term_months"] = (
                df["term"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

        if "earliest_cr_line" in df.columns:
            earliest_credit = pd.to_datetime(
                df["earliest_cr_line"],
                format="%b-%Y",
                errors="coerce"
            )

            reference_date = pd.Timestamp("2018-12-31")
            df["credit_history_months"] = (
                (reference_date - earliest_credit).dt.days / 30
            )

        if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
            df["fico_avg"] = (
                pd.to_numeric(df["fico_range_low"], errors="coerce")
                + pd.to_numeric(df["fico_range_high"], errors="coerce")
            ) / 2

        for col in self.required_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.required_columns]

        return df

    def predict(self, input_data):
        processed_data = self.preprocess_input(input_data)

        prob = self.pipeline.predict_proba(processed_data)[0][1]
        pred = self.pipeline.predict(processed_data)[0]

        prediction = "Default" if pred == 1 else "Fully Paid"

        return {
            "prediction": prediction,
            "default_probability": round(float(prob), 4)
        }


if __name__ == "__main__":
    agent = LoanDefaultAgent()

    sample_input = {
        "loan_amnt": 5000,
        "funded_amnt": 5000,
        "funded_amnt_inv": 5000,
        "term": "36 months",
        "int_rate": "10.5%",
        "installment": 162.49,
        "grade": "B",
        "sub_grade": "B1",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 20,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Jan-2005",
        "fico_range_low": 715,
        "fico_range_high": 720,
        "inq_last_6mths": 1,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 5000,
        "revol_util": "30%",
        "total_acc": 20,
        "initial_list_status": "w",
        "application_type": "Individual",
        "mort_acc": 2,
        "pub_rec_bankruptcies": 0
    }

    result = agent.predict(sample_input)
    print(f"\nPrediction Result: {result}")