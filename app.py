"""
app.py
------
Interactive Streamlit Web Dashboard for Disease Prediction from Medical Data.

Features:
1. Multi-disease selection (Breast Cancer, Diabetes, Heart Disease)
2. Choice of 4 ML Algorithms (Logistic Regression, SVM, Random Forest, XGBoost)
3. Custom patient vital inputs (Age, Glucose, Blood Pressure, BMI, Heart Rate, etc.)
4. Instant Disease Risk Prediction with Probability Score & Visual Metrics
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from data_loader import get_preprocessed_dataset, load_breast_cancer_data, load_diabetes_data, load_heart_disease_data
from train_evaluate import get_models

# Page Config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


def ensure_models_trained(dataset_name):
    """
    Ensures model and scaler exist on disk, trains them if missing.
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    scaler_path = os.path.join(models_dir, f"{dataset_name}_scaler.pkl")
    features_path = os.path.join(models_dir, f"{dataset_name}_features.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, X_raw = get_preprocessed_dataset(dataset_name)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, features_path)

        models = get_models()
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            model_path = os.path.join(models_dir, f"{dataset_name}_{name.replace(' ', '_').lower()}.pkl")
            joblib.dump(model, model_path)


def main():
    st.markdown("<div class='main-header'>🩺 AI Disease Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict clinical disease outcomes using machine learning classification algorithms</div>", unsafe_allow_html=True)

    # Sidebar Selection Controls
    st.sidebar.header("⚙️ Configuration Panel")

    dataset_option = st.sidebar.selectbox(
        "1. Select Target Disease",
        ["Breast Cancer", "Diabetes", "Heart Disease"],
        index=1
    )

    dataset_key = dataset_option.lower().replace(" ", "_")
    ensure_models_trained(dataset_key)

    algo_option = st.sidebar.selectbox(
        "2. Select Classification Algorithm",
        ["Logistic Regression", "SVM", "Random Forest", "XGBoost"],
        index=2
    )

    # Load pre-trained model & scaler
    models_dir = "models"
    model_path = os.path.join(models_dir, f"{dataset_key}_{algo_option.replace(' ', '_').lower()}.pkl")
    scaler_path = os.path.join(models_dir, f"{dataset_key}_scaler.pkl")
    features_path = os.path.join(models_dir, f"{dataset_key}_features.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    # Main Layout
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"📋 Patient Medical Input ({dataset_option})")
        st.markdown("Enter patient physiological parameters & lab results below:")

        user_inputs = {}

        # Custom inputs based on disease type
        if dataset_key == "breast_cancer":
            c1, c2 = st.columns(2)
            with c1:
                user_inputs['mean radius'] = st.number_input("Mean Radius (mm)", 6.0, 30.0, 14.0, 0.1)
                user_inputs['mean texture'] = st.number_input("Mean Texture", 9.0, 40.0, 19.0, 0.1)
                user_inputs['mean perimeter'] = st.number_input("Mean Perimeter (mm)", 40.0, 190.0, 92.0, 0.1)
                user_inputs['mean area'] = st.number_input("Mean Area (mm²)", 140.0, 2500.0, 650.0, 1.0)
                user_inputs['mean smoothness'] = st.number_input("Mean Smoothness", 0.05, 0.20, 0.10, 0.01)
            with c2:
                user_inputs['mean compactness'] = st.number_input("Mean Compactness", 0.01, 0.35, 0.10, 0.01)
                user_inputs['mean concavity'] = st.number_input("Mean Concavity", 0.0, 0.45, 0.09, 0.01)
                user_inputs['mean concave points'] = st.number_input("Mean Concave Points", 0.0, 0.20, 0.05, 0.01)
                user_inputs['mean symmetry'] = st.number_input("Mean Symmetry", 0.10, 0.35, 0.18, 0.01)
                user_inputs['mean fractal dimension'] = st.number_input("Mean Fractal Dim", 0.04, 0.10, 0.06, 0.005)

        elif dataset_key == "diabetes":
            c1, c2 = st.columns(2)
            with c1:
                user_inputs['Pregnancies'] = st.slider("Pregnancies", 0, 17, 1)
                user_inputs['Glucose'] = st.slider("Glucose Level (mg/dL)", 40, 200, 120)
                user_inputs['BloodPressure'] = st.slider("Blood Pressure (mmHg)", 40, 140, 70)
                user_inputs['SkinThickness'] = st.slider("Skin Thickness (mm)", 0, 99, 20)
            with c2:
                user_inputs['Insulin'] = st.slider("Insulin Level (mu U/ml)", 0, 846, 80)
                user_inputs['BMI'] = st.slider("Body Mass Index (BMI)", 15.0, 60.0, 28.5, 0.1)
                user_inputs['DiabetesPedigreeFunction'] = st.slider("Diabetes Pedigree Function", 0.05, 2.50, 0.47, 0.01)
                user_inputs['Age'] = st.slider("Age (years)", 18, 100, 33)

        elif dataset_key == "heart_disease":
            c1, c2 = st.columns(2)
            with c1:
                user_inputs['age'] = st.slider("Age", 20, 90, 54)
                user_inputs['sex'] = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
                user_inputs['cp'] = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
                user_inputs['trestbps'] = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
                user_inputs['chol'] = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)
            with c2:
                user_inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")
                user_inputs['restecg'] = st.selectbox("Resting ECG Results", options=[0, 1, 2])
                user_inputs['thalach'] = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
                user_inputs['exang'] = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                user_inputs['oldpeak'] = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)

        predict_clicked = st.button("🔬 RUN DISEASE PREDICTION")

    with col_right:
        st.subheader("📊 Prediction Results & Analytics")

        if predict_clicked:
            # Prepare feature vector matching features list order
            input_values = [user_inputs[f] for f in feature_names]
            input_df = pd.DataFrame([input_values], columns=feature_names)

            # Scale inputs
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else 0.5

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.error(f"### ⚠️ HIGH RISK OF {dataset_option.upper()}")
                st.markdown(f"**Model Confidence:** `{probability * 100:.1f}%`")
                st.progress(float(probability))
                st.warning("Recommendation: Clinical follow-up and further laboratory evaluation are advised.")
            else:
                st.success(f"### ✅ LOW RISK / HEALTHY ({dataset_option.upper()})")
                st.markdown(f"**Model Confidence (Healthy):** `{(1 - probability) * 100:.1f}%`")
                st.progress(float(1 - probability))
                st.info("Recommendation: Patient metrics fall within normal reference ranges.")

            st.divider()

        # Display Model Performance Metrics Table
        results_csv = os.path.join(models_dir, f"{dataset_key}_results.csv")
        if os.path.exists(results_csv):
            st.markdown(f"#### 🏆 Model Performance Comparison on {dataset_option}")
            res_df = pd.read_csv(results_csv)
            st.dataframe(res_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score', 'ROC-AUC'], color='#D1FAE5'), use_container_width=True)

        # Display saved plot if available
        comp_plot = os.path.join("plots", f"{dataset_key}_model_comparison.png")
        if os.path.exists(comp_plot):
            st.image(comp_plot, caption=f"{dataset_option} Model Evaluation Metrics Chart", use_container_width=True)


if __name__ == "__main__":
    main()
