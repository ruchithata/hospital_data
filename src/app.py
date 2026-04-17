import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import os

# ===============================
# 🎨 PAGE CONFIG & CUSTOM CSS
# ===============================
st.set_page_config(page_title="MedForecast AI", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek typography */
    h1, h2, h3 {
        font-weight: 700 !important;
        color: #1f2937;
    }
    
    /* Metrics Box Stylish Design */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2563eb;
    }

    /* Clean up top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# 🏥 HEADER 
# ===============================
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; margin-bottom: 2rem; background: linear-gradient(to right, #eff6ff, #dbeafe); border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>
        <h1 style='color: #1e3a8a; margin-bottom: 0.5rem; font-size: 3rem;'>🏥 MedForecast AI</h1>
        <p style='color: #475569; font-size: 1.2rem; margin: 0;'>Smart Hospital Resource Planning & Predictive Analytics</p>
    </div>
""", unsafe_allow_html=True)

# ===============================
# 📂 DATA UPLOAD (SIDEBAR)
# ===============================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("Upload hospital data or use the defaults to start predicting resource needs.")
    
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"], help="Format: day, admissions, beds_used, oxygen_used")

st.sidebar.markdown("---")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded!")
else:
    # Try reading default dataset gracefully
    default_path = "../data/data.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.info("ℹ️ Using default dataset")
    elif os.path.exists("./data/data.csv"):
        df = pd.read_csv("./data/data.csv")
        st.sidebar.info("ℹ️ Using default dataset")
    else: 
        st.error("Could not find default dataset at `../data/data.csv`. Please upload one.")
        st.stop()

# ===============================
# ⚠️ VALIDATION
# ===============================
required_columns = ["day", "admissions", "beds_used", "oxygen_used"]
if not all(col in df.columns for col in required_columns):
    st.error(f"Dataset must contain these columns: {', '.join(required_columns)}")
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
# 📌 NAVIGATION (TOP TABS)
# ===============================
tab1, tab2 = st.tabs(["🔮 Real-Time Prediction", "📈 Advanced Analytics"])

# ===============================
# 🔮 TAB 1: PREDICTION
# ===============================
with tab1:
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("### 🎛️ Simulation Parameters")
        st.markdown("Adjust the expected hospital admissions to predict resource scaling requirements.")
        
        # Default value dynamically based on average
        default_admissions = int(df["admissions"].mean())
        admissions = st.slider("Patients Expected", min_value=0, max_value=int(max(300, df["admissions"].max()+100)), value=default_admissions, step=5)
        
        st.info("💡 Predictions update instantly based on the slider value.")

    input_data = pd.DataFrame([[admissions]], columns=["admissions"])
    beds_pred = model_beds.predict(input_data)[0]
    oxygen_pred = model_oxygen.predict(input_data)[0]

    with col2:
        st.markdown("### ⚡ Resource Forecast")
        
        colA, colB = st.columns(2)
        with colA:
            st.metric(label="🛏️ Beds Needed", value=f"{beds_pred:.1f}", delta=f"{(beds_pred - df['beds_used'].mean()):.1f} vs Avg", delta_color="inverse")
        with colB:
            st.metric(label="🫧 Oxygen Needed (L)", value=f"{oxygen_pred:.1f}", delta=f"{(oxygen_pred - df['oxygen_used'].mean()):.1f} vs Avg", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        # Dynamic alert based on threshold
        if beds_pred > 70:
            st.error("🚨 **High Demand Expected!** Prepare additional beds and check oxygen inventory immediately.", icon="⚠️")
        else:
            st.success("✅ **Resources Sufficient.** The facility is operating within normal parameters.", icon="🟢")

# ===============================
# 📈 TAB 2: ANALYTICS
# ===============================
with tab2:
    st.markdown("### 📊 Interactive Resource Insights")
    st.markdown("Explore historical data patterns to understand how admissions impact resource consumption.")

    # Generate a plot to show trends
    fig1 = px.line(df, x="day", y=["beds_used", "oxygen_used"],
                   labels={"value": "Quantity", "variable": "Resource Type"},
                   title="Historical Resource Utilisation",
                   color_discrete_sequence=["#3b82f6", "#10b981"])
    fig1.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Generate scatter plot
    fig2 = px.scatter(df, x="admissions", y="beds_used", size="oxygen_used", color="admissions",
                      hover_data=["day"], color_continuous_scale="Viridis",
                      title="Admissions vs Beds (Bubble Size = Oxygen)")
    
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.plotly_chart(fig1, use_container_width=True)
    with viz_col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    data_col1, data_col2 = st.columns([1, 1])
    with data_col1:
        st.markdown("### 📋 Deep Dive Dataset")
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with data_col2:
        st.markdown("### 🧮 Statistical Summary")
        st.dataframe(df.describe().T[['mean', 'min', 'max']], use_container_width=True)
