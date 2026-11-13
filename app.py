"""Streamlit interface for the heart-disease classifier.

This application is an educational demonstration. It must not be used to
diagnose, treat, or replace advice from a qualified health-care professional.
"""

import joblib
import pandas as pd
import streamlit as st

from pathlib import Path
from preprocessing import RAW_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_pipeline.pkl"


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model(model_path: Path):
    """Load the locally packaged model.

    Joblib/pickle files can execute code when deserialized, so only the model
    packaged with this project should ever be placed at this path.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    return joblib.load(model_path)


def make_input_frame(values: dict[str, int | float]) -> pd.DataFrame:
    """Return one raw-input row for the complete preprocessing/model pipeline."""
    frame = pd.DataFrame([[values[column] for column in RAW_FEATURE_COLUMNS]], columns=RAW_FEATURE_COLUMNS)
    if frame.isna().any().any():
        raise ValueError("All health parameters are required.")
    return frame


def main() -> None:
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
    st.markdown("""
        <style>
        .stApp { background: #0e1117; }
        .block-container { max-width: 720px; padding-top: 2.4rem; padding-bottom: 3rem; }
        .app-title { font-size: 2.15rem; font-weight: 750; margin: 0 0 1.2rem; }
        .patient-card { border: 1px solid #3b404b; border-radius: 9px; padding: 1.35rem 1rem .9rem; margin-bottom: 1rem; }
        .patient-card h2 { font-size: 1.35rem; margin: 0 0 .8rem; }
        div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label { font-size: .9rem; font-weight: 600; }
        div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div { background: #262730; border-color: #262730; }
        div[data-testid="stFormSubmitButton"] > button { background: #141820; border: 1px solid #555c68; color: #f5f5f5; border-radius: 6px; font-weight: 600; }
        .result-card { border-radius: 8px; padding: .95rem 1rem; font-weight: 600; }
        .result-risk { background: #482a2e; color: #ffb7b7; }
        .result-low { background: #193d2a; color: #b8eccb; }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<div class="app-title">&#10084;&#65039; Heart Disease Prediction App</div>', unsafe_allow_html=True)
    st.caption("Educational machine-learning demonstration")
    st.warning(
        "Medical disclaimer: this tool is not a diagnosis and must not guide medical "
        "decisions. Discuss symptoms, screening, and treatment with a qualified clinician. "
        "Seek emergency care for urgent symptoms."
    )

    try:
        model = load_model(MODEL_PATH)
    except Exception as error:
        st.error(f"The local model could not be loaded: {error}")
        st.stop()

    st.markdown('<div class="patient-card"><h2>Enter Patient Details</h2>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        age = st.number_input("Age", 1, 120, 45, 1)
        sex = st.selectbox("Sex (1=male, 0=female)", (0, 1), format_func=str)
        cp = st.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120, 1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 50, 700, 200, 1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=yes, 0=no)", (0, 1))
        restecg = st.selectbox("Resting ECG Results (0-2)", (0, 1, 2))
        thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150, 1)
        exang = st.selectbox("Exercise Induced Angina (1=yes, 0=no)", (0, 1))
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("Slope (0=up, 1=flat, 2=down)", (0, 1, 2))
        ca = st.selectbox("Number of Major Vessels (0-3)", (0, 1, 2, 3, 4))
        thal = st.selectbox("Thal value (1-3)", (0, 1, 2, 3))
        submitted = st.form_submit_button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    values = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    try:
        input_frame = make_input_frame(values)
        prediction = model.predict(input_frame)[0]
    except (ValueError, TypeError) as error:
        st.error(f"Please correct the input values: {error}")
        return
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        return

    probability = None
    if callable(getattr(model, "predict_proba", None)):
        probabilities = model.predict_proba(input_frame)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            probability = probabilities[classes.index(1)]

    if prediction == 1:
        result = "High Risk of Heart Disease"
        if probability is not None:
            result += f" (Confidence: {probability:.2%})"
        st.markdown(f'<div class="result-card result-risk">{result}</div>', unsafe_allow_html=True)
    else:
        result = "Lower Risk of Heart Disease"
        if probability is not None:
            result += f" (Confidence: {1 - probability:.2%})"
        st.markdown(f'<div class="result-card result-low">{result}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
