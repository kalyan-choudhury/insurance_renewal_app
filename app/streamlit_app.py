import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ===============================
# ⚙️ PAGE CONFIGURATION
# ===============================
st.set_page_config(page_title="Insurance Renewal Prediction", page_icon="💼", layout="wide")
st.title("💼 Insurance Renewal Prediction App")
st.write("Upload customer data to predict the probability of renewal.")

# ===============================
# 📦 LOAD TRAINED MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("models/final_model.pkl")

model = load_model()

# ===============================
# 🧠 PREPROCESSING FUNCTION
# ===============================
def preprocess_input(df):
    # Handle Missing Values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Late payment ratios
    total_premiums = (
        df["no_of_premiums_paid"]
        + df["Count_3-6_months_late"]
        + df["Count_6-12_months_late"]
        + df["Count_more_than_12_months_late"]
    )
    df["late_ratio_3_6"] = df["Count_3-6_months_late"] / total_premiums
    df["late_ratio_6_12"] = df["Count_6-12_months_late"] / total_premiums
    df["late_ratio_more_12"] = df["Count_more_than_12_months_late"] / total_premiums

    # Age bucket
    df["age_bucket"] = pd.cut(
        (df["age_in_days"] / 365).astype(int),
        bins=[20, 30, 40, 50, 60, 70, 100],
        labels=["20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    )

    # Drop unnecessary columns
    for c in ["id", "age_in_days"]:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True, errors='ignore')

    # One-hot encode categorical variables
    df = pd.get_dummies(
        df,
        columns=["residence_area_type", "sourcing_channel", "age_bucket"],
        drop_first=True
    )

    # Scale numeric columns (recreate scaler)
    scaler = StandardScaler()
    num_cols = [c for c in ["Income", "application_underwriting_score"] if c in df.columns]
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Drop unused columns if present
    drop_cols = ["Count_3-6_months_late", "Count_6-12_months_late", "Count_more_than_12_months_late"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True, errors='ignore')

    # Align columns with model expectations if possible
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]

    return df


# ===============================
# 📤 FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Input Preview")
    st.dataframe(data.head())

    try:
        processed = preprocess_input(data)
        st.subheader("⚙️ Processed Data Sample")
        st.dataframe(processed.head())

        preds = model.predict(processed)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(processed)[:, 1]
        else:
            prob = preds

        # Combine original input and predictions
        results = data.copy()
        results["Predicted_Renewal"] = preds
        results["Renewal_Probability"] = prob

        st.subheader("🔮 Predictions with Input Data")
        st.dataframe(results.head())

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=results.to_csv(index=False),
            file_name="predictions.csv"
        )


    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

else:
    st.info("Please upload a CSV file with customer data to begin.")
