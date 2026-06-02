import os
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

warnings.filterwarnings("ignore")

DATA_PATH = "accepted_2007_to_2018Q4.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150)
    plt.close()


def clean_percent(series):
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["nan", "None", ""], np.nan)
        .astype(float)
    )


def safe_category_order(series):
    return sorted(series.dropna().astype(str).unique())


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


print("--- Phase 1: Loading Data ---")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Original shape: {df.shape}")


print("\n--- Phase 2: Target Creation ---")

if "loan_status" not in df.columns:
    raise KeyError("loan_status column is required.")

status_map = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

df["target"] = df["loan_status"].map(status_map)
df = df[df["target"].isin([0, 1])].copy()

if df.empty:
    raise ValueError("No valid target rows found.")

print(df["target"].value_counts())


print("\n--- Phase 3: Select Safe Pre-Loan Features ---")

safe_features = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

available_features = [col for col in safe_features if col in df.columns]
df = df[available_features + ["target"]].copy()

print(f"Selected feature count: {len(available_features)}")
print(f"Shape after feature selection: {df.shape}")


print("\n--- Phase 4: Cleaning Feature Types ---")

for col in ["int_rate", "revol_util"]:
    if col in df.columns:
        df[col] = clean_percent(df[col])

if "term" in df.columns:
    df["term_months"] = (
        df["term"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )
    df = df.drop(columns=["term"])

if "earliest_cr_line" in df.columns:
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce"
    )
    reference_date = pd.Timestamp("2018-12-31")
    df["credit_history_months"] = (
        (reference_date - df["earliest_cr_line"]).dt.days / 30
    )
    df = df.drop(columns=["earliest_cr_line"])

if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])

for col in df.select_dtypes(include=["object"]).columns:
    if col != "target":
        df[col] = df[col].astype(str).replace("nan", np.nan)


print("\n--- Phase 5: EDA Plots ---")

if "grade" in df.columns and "loan_amnt" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="grade",
        y="loan_amnt",
        data=df,
        order=safe_category_order(df["grade"])
    )
    plt.title("Loan Amount by Grade")
    save_plot("loan_amount_by_grade.png")

if "loan_amnt" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["loan_amnt"].dropna(), bins=40, kde=True)
    plt.title("Loan Amount Distribution")
    save_plot("loan_amount_distribution.png")

if "int_rate" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["int_rate"].dropna(), bins=40, kde=True)
    plt.title("Interest Rate Distribution")
    save_plot("interest_rate_distribution.png")

if "annual_inc" in df.columns:
    income = df.loc[df["annual_inc"] > 0, "annual_inc"].dropna()
    if not income.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(income, bins=40, kde=True)
        plt.xscale("log")
        plt.title("Annual Income Distribution")
        save_plot("annual_income_distribution.png")


print("\n--- Phase 6: Prepare Train/Test Data ---")

X = df.drop(columns=["target"])
y = df["target"].astype(int)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

if y.nunique() < 2:
    raise ValueError("Training requires both classes: 0 and 1.")

can_stratify = y.value_counts().min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y if can_stratify else None,
)


print("\n--- Phase 7: Build Preprocessing Pipeline ---")

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ]
)


print("\n--- Phase 8: Train Models ---")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "auc": None,
        "pipeline": pipeline,
    }

    if len(np.unique(y_test)) == 2 and hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        result["auc"] = roc_auc_score(y_test, y_prob)

    results[name] = result


print("\n--- Phase 9: Evaluation ---")

best_model_name = None
best_auc = -1

for name, result in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {result['accuracy']:.4f}")

    if result["auc"] is not None:
        print(f"AUC-ROC: {result['auc']:.4f}")
        if result["auc"] > best_auc:
            best_auc = result["auc"]
            best_model_name = name
    else:
        print("AUC-ROC: Not available")

    print("Classification Report:")
    print(result["report"])

if best_model_name is None:
    best_model_name = max(results, key=lambda name: results[name]["accuracy"])

best_pipeline = results[best_model_name]["pipeline"]

joblib.dump(best_pipeline, "loan_default_pipeline.pkl")
joblib.dump(available_features, "selected_features.pkl")

print(f"\nBest model saved: {best_model_name}")
print("Saved file: loan_default_pipeline.pkl")
print("Pipeline complete.")