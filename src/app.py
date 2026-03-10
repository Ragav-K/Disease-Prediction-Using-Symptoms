import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="centered")

@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    et = joblib.load(os.path.join(MODELS_DIR, "extra_trees.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return rf, et, le

@st.cache_data
def load_symptoms():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "processed", "X_train.csv"))
    return list(X_train.columns)

rf, et, le = load_models()
SYMPTOMS = load_symptoms()

st.title("🩺 Disease Prediction System")
st.markdown("Select your symptoms below and click **Predict** to get results.")
st.markdown("---")

selected_symptoms = st.multiselect(
    "🔍 Search and select your symptoms:",
    options=SYMPTOMS,
    placeholder="Type to search symptoms..."
)

st.markdown(f"**Selected:** {len(selected_symptoms)} symptom(s)")

if st.button("🔮 Predict Disease", use_container_width=True, type="primary"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom before predicting.")
    else:
        input_vector = np.zeros(len(SYMPTOMS))
        for symptom in selected_symptoms:
            input_vector[SYMPTOMS.index(symptom)] = 1

        input_df = pd.DataFrame([input_vector], columns=SYMPTOMS)

        with st.spinner("Analyzing symptoms..."):
            rf_proba = rf.predict_proba(input_df)
            et_proba = et.predict_proba(input_df)
            avg_proba = (rf_proba + et_proba) / 2

        top3_idx = np.argsort(avg_proba[0])[::-1][:3]
        top3_diseases = le.inverse_transform(top3_idx)
        top3_raw = avg_proba[0][top3_idx]
        top3_scores = top3_raw / top3_raw.sum()

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        st.success(f"🏆 **Most Likely Disease:** {top3_diseases[0]}")
        st.progress(float(top3_scores[0]))
        st.caption(f"Confidence: {top3_scores[0]*100:.1f}%")

        st.markdown("**Other Possibilities:**")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"2️⃣ {top3_diseases[1]}")
            st.caption(f"{top3_scores[1]*100:.1f}% confidence")
        with col2:
            st.info(f"3️⃣ {top3_diseases[2]}")
            st.caption(f"{top3_scores[2]*100:.1f}% confidence")

        st.markdown("---")
        st.caption("⚠️ For informational purposes only. Consult a qualified doctor for medical advice.")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Ensemble ML model (Random Forest + Extra Trees) trained on a medical symptom dataset.")
    st.markdown("---")
    st.markdown(f"**Total Symptoms Available:** {len(SYMPTOMS)}")
    st.markdown(f"**Disease Classes:** {len(le.classes_)}")
    st.markdown(f"**Model Accuracy:** ~80.7%")
    st.markdown("**Models:** Random Forest + Extra Trees (Soft Voting)")
