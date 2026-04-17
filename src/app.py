import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ===============================
# 🏥 TITLE
# ===============================
st.title("🏥 MedForecast AI – Hospital Resource Planner")

# ===============================
# 📌 SIDEBAR NAVIGATION
# ===============================
page = st.sidebar.radio("📌 Navigate", ["🏠 Prediction", "📊 Data & Visualization"])

# ===============================
# 📂 DATA UPLOAD
# ===============================
st.sidebar.markdown("### 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("../data/data.csv")

# ===============================
# ⚠️ VALIDATION
# ===============================
required_columns = ["day", "admissions", "beds_used", "oxygen_used"]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset must contain: day, admissions, beds_used, oxygen_used")
    st.stop()

# ===============================
# 🤖 MODEL TRAINING
# ===============================
X = df[["admissions"]]

model_beds = LinearRegression()
model_beds.fit(X, df["beds_used"])

model_oxygen = LinearRegression()
model_oxygen.fit(X, df["oxygen_used"])

# ===============================
# 🏠 PAGE 1: PREDICTION
# ===============================
if page == "🏠 Prediction":

    st.header("📊 Resource Prediction")

    # Input
    admissions = st.slider("Expected Admissions", 0, 200, 80)

    input_data = pd.DataFrame([[admissions]], columns=["admissions"])

    # Predictions
    beds_pred = model_beds.predict(input_data)
    oxygen_pred = model_oxygen.predict(input_data)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🛏 Beds Needed", f"{beds_pred[0]:.2f}")

    with col2:
        st.metric("🫧 Oxygen Needed", f"{oxygen_pred[0]:.2f}")

    # Alert
    if beds_pred[0] > 70:
        st.error("⚠️ High demand expected! Prepare additional resources.")
    else:
        st.success("✅ Resources are sufficient.")

# ===============================
# 📊 PAGE 2: VISUALIZATION
# ===============================
elif page == "📊 Data & Visualization":

    st.header("📈 Data Insights & Visualization")

    # Input for dynamic graph
    admissions = st.slider("Simulate Admissions", 0, 200, 80)

    input_data = pd.DataFrame([[admissions]], columns=["admissions"])

    beds_pred = model_beds.predict(input_data)
    oxygen_pred = model_oxygen.predict(input_data)

    # Add predicted row
    new_day = df["day"].max() + 1

    new_row = pd.DataFrame({
        "day": [new_day],
        "admissions": [admissions],
        "beds_used": [beds_pred[0]],
        "oxygen_used": [oxygen_pred[0]]
    })

    df_extended = pd.concat([df, new_row], ignore_index=True)

    # Graphs
    st.subheader("📈 Resource Usage Trend")
    st.line_chart(df_extended.set_index("day")[["beds_used", "oxygen_used"]])

    st.subheader("🔵 Admissions vs Beds")
    st.scatter_chart(df_extended[["admissions", "beds_used"]])

    st.subheader("📊 Bed Usage")
    st.bar_chart(df_extended.set_index("day")["beds_used"])

    st.caption("Last data point represents predicted future value")

    # Data preview
    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())