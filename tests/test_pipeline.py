"""Regression tests for the locally packaged complete model pipeline."""

from pathlib import Path
import unittest

import joblib
import pandas as pd

from preprocessing import PROCESSED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parents[1]


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_accepts_raw_fields_and_matches_processed_schema(self):
        pipeline = joblib.load(BASE_DIR / "models" / "heart_disease_pipeline.pkl")
        raw = pd.read_csv(BASE_DIR / "data" / "heart.csv").drop_duplicates().drop(columns="target")
        sample = raw.loc[:, RAW_FEATURE_COLUMNS].head(1)
        transformed = pipeline.named_steps["preprocessor"].transform(sample)
        self.assertEqual(tuple(transformed.columns), PROCESSED_FEATURE_COLUMNS)
        self.assertEqual(transformed.shape, (1, 22))
        self.assertEqual(len(pipeline.predict(sample)), 1)
        self.assertEqual(len(pipeline.predict_proba(sample)[0]), 2)


if __name__ == "__main__":
    unittest.main()
