"""Build the complete raw-input pipeline from the supplied processed dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import HeartFeaturePreprocessor, PROCESSED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "heart.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "heart_cleaned_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_pipeline.pkl"
TARGET_COLUMN = "target"


def train_and_evaluate() -> dict[str, object]:
    """Train the Untitled notebook's Random Forest as one raw-input pipeline."""
    raw_dataset = pd.read_csv(RAW_DATA_PATH).drop_duplicates().reset_index(drop=True)
    processed_dataset = pd.read_csv(PROCESSED_DATA_PATH)
    if tuple(processed_dataset.columns) != (*PROCESSED_FEATURE_COLUMNS, TARGET_COLUMN):
        raise ValueError("Processed CSV schema does not match the SageMaker training schema.")
    if len(raw_dataset) != len(processed_dataset) or not raw_dataset[TARGET_COLUMN].equals(processed_dataset[TARGET_COLUMN]):
        raise ValueError("Raw and processed datasets are not aligned; refusing to build a pipeline.")

    raw_predictors = raw_dataset.loc[:, RAW_FEATURE_COLUMNS]
    preprocessor = HeartFeaturePreprocessor().fit(raw_predictors)
    reconstructed = preprocessor.transform(raw_predictors)
    expected = processed_dataset.loc[:, PROCESSED_FEATURE_COLUMNS]
    if not np.allclose(reconstructed.to_numpy(), expected.to_numpy(), atol=5e-5, rtol=0):
        raise ValueError("Reconstructed preprocessing does not reproduce the supplied processed CSV.")

    x_train, x_test, y_train, y_test = train_test_split(
        expected, processed_dataset[TARGET_COLUMN], test_size=0.20, random_state=42,
        stratify=processed_dataset[TARGET_COLUMN],
    )
    classifier = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        random_state=42, class_weight="balanced",
    ).fit(x_train, y_train)
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    predictions = model.predict(raw_predictors.loc[x_test.index])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "features": list(PROCESSED_FEATURE_COLUMNS),
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    for name, value in train_and_evaluate().items():
        print(f"{name}: {value}")
