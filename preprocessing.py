"""Feature preparation shared by training and the Streamlit application."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_FEATURE_COLUMNS = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
)

PROCESSED_FEATURE_COLUMNS = (
    "age", "sex", "resting_bp", "cholesterol", "fasting_bs_gt120",
    "max_heart_rate", "exercise_angina", "st_depression", "major_vessels",
    "cp_0", "cp_1", "cp_2", "cp_3", "ecg_0", "ecg_1", "ecg_2",
    "slope_0", "slope_1", "slope_2", "thal_1", "thal_2", "thal_3",
)

_SCALED_COLUMNS = ("age", "trestbps", "chol", "thalach", "oldpeak")


class HeartFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Convert the website's 13 raw fields into the 22 trained features.

    The scaler is fitted only during model training.  Its learned values are
    serialized in the complete pipeline along with the classifier.
    """

    def fit(self, x: pd.DataFrame, y=None):
        frame = self._validate_frame(x)
        self.feature_names_in_ = np.asarray(RAW_FEATURE_COLUMNS, dtype=object)
        # The supplied processed CSV uses pandas' sample standard deviation
        # (ddof=1), not sklearn StandardScaler's population deviation.
        self.means_ = frame.loc[:, _SCALED_COLUMNS].mean()
        self.scales_ = frame.loc[:, _SCALED_COLUMNS].std(ddof=1)
        if (self.scales_ == 0).any():
            raise ValueError("A numeric feature has zero variance.")
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "means_"):
            raise RuntimeError("The feature preprocessor has not been fitted.")
        frame = self._validate_frame(x)
        scaled = ((frame.loc[:, _SCALED_COLUMNS] - self.means_) / self.scales_).to_numpy()
        output = pd.DataFrame(index=frame.index)
        output["age"] = scaled[:, 0]
        output["sex"] = frame["sex"].astype(int)
        output["resting_bp"] = scaled[:, 1]
        output["cholesterol"] = scaled[:, 2]
        output["fasting_bs_gt120"] = frame["fbs"].astype(int)
        output["max_heart_rate"] = scaled[:, 3]
        output["exercise_angina"] = frame["exang"].astype(int)
        output["st_depression"] = scaled[:, 4]
        # The cleaned source maps the four records coded as ca=4 to 0.
        output["major_vessels"] = frame["ca"].where(frame["ca"].between(0, 3), 0).astype(int)
        for column, values in (("cp", range(4)), ("restecg", range(3)), ("slope", range(3)), ("thal", range(1, 4))):
            source = frame[column]
            # The two thal=0 records in the raw source were cleaned to the
            # mode (2) before one-hot encoding in the supplied dataset.
            if column == "thal":
                source = source.where(source != 0, 2)
            for value in values:
                output[f"{ {'restecg': 'ecg'}.get(column, column) }_{value}"] = (
                    source == value
                ).astype(int)
        return output.loc[:, PROCESSED_FEATURE_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(PROCESSED_FEATURE_COLUMNS, dtype=object)

    @staticmethod
    def _validate_frame(x: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with the 13 raw feature columns.")
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in x.columns]
        unexpected = [column for column in x.columns if column not in RAW_FEATURE_COLUMNS]
        if missing or unexpected:
            raise ValueError(f"Raw feature schema mismatch; missing={missing}, unexpected={unexpected}.")
        frame = x.loc[:, RAW_FEATURE_COLUMNS].copy()
        if frame.isna().any().any():
            raise ValueError("All health parameters are required.")
        return frame
