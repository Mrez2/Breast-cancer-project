import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer

st.title("🩺 Breast Cancer Tumor Classification")

# -----------------------
# Ensure session_state key exists (but don't reset it!)
# -----------------------
if "svm_model" not in st.session_state:
    st.session_state.svm_model = None

# -----------------------
# Load dataset to get feature names & means
# -----------------------
data = load_breast_cancer()
feature_names = data.feature_names  # 30 features
feature_means = data.data.mean(axis=0)  # mean of each feature

# Map features
feature_index_map = {name: i for i, name in enumerate(feature_names)}

# -----------------------
# Top 10 important features
# -----------------------
important_features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error"
]

# Defaults from first row
feature_defaults = {
    f: float(data.data[0][feature_index_map[f]]) for f in important_features
}

# -----------------------
# Model loading
# -----------------------
if st.button("Load Best Model (SVM)"):
    try:
        with open("SVM.pkl", "rb") as f:
            st.session_state.svm_model = pickle.load(f)
        st.success("✅ SVM model loaded successfully!")
    except FileNotFoundError:
        st.error("❌ SVM model file not found. Please train and save it first.")

# Show status of model (always visible)
if st.session_state.svm_model is not None:
    st.info(f"📦 Model is loaded. It expects {st.session_state.svm_model.n_features_in_} features.")
else:
    st.warning("⚠️ No model loaded yet.")

# -----------------------
# User input
# -----------------------
st.header("🧪 Enter Tumor Features (Top 10 Important Features)")
user_input = {}
for feature in important_features:
    user_input[feature] = st.number_input(feature, value=feature_defaults[feature])

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Tumor Type"):
    if st.session_state.svm_model is None:
        st.warning("⚠️ Please load the SVM model first!")
    else:
        full_features = []
        for idx, f in enumerate(feature_names):
            if f in important_features:
                full_features.append(user_input[f])
            else:
                full_features.append(feature_means[idx])

        full_features = np.array(full_features).reshape(1, -1)

        # Pad if model expects more
        expected_features = st.session_state.svm_model.n_features_in_
        if full_features.shape[1] < expected_features:
            diff = expected_features - full_features.shape[1]
            full_features = np.hstack([full_features, np.zeros((1, diff))])

        try:
            prediction = st.session_state.svm_model.predict(full_features)
            if prediction[0] == 0:
                st.success("✅ Prediction: Benign (B)")
            else:
                st.error("⚠️ Prediction: Malignant (M)")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
