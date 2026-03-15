import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg,#1F4E79,#4A90E2);
}
.main-container {
    background-color: rgba(255,255,255,0.95);
    padding:40px;
    border-radius:18px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.2);
}
.title {
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:#1F4E79;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:40px;
}
.section-title{
    font-size:26px;
    font-weight:700;
    margin-bottom:15px;
    color:#1F4E79;
}
.prediction-card {
    background: linear-gradient(135deg,#43cea2,#185a9d);
    padding:30px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    color:white;
    margin-top:20px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.2);
}
.stButton>button {
    background: linear-gradient(135deg,#1F4E79,#4A90E2);
    color:white;
    font-size:20px;
    font-weight:600;
    border-radius:12px;
    height:55px;
    width:100%;
    border:none;
}
.stButton>button:hover{
    transform:scale(1.02);
    background: linear-gradient(135deg,#154360,#1F4E79);
}
.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    """
    Load compressed joblib model from models folder.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "car_price_model_compressed.pkl")

    if not os.path.exists(model_path):
        st.error("Model file not found! Please upload 'car_price_model_compressed.pkl' in models folder.")
        st.stop()

    package = joblib.load(model_path)
    return package

# Load package
package = load_model()
model = package["model"]
label_encoders = package["label_encoders"]
selected_features = package["selected_features"]

# ---------------- PAGE HEADER ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<p class="title">🚗 AI Car Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict resale value of a car using Machine Learning (Random Forest)</p>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('<p class="section-title">Enter Car Details</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox("Manufacturer", label_encoders["Manufacturer"].classes_)
    category = st.selectbox("Category", label_encoders["Category"].classes_)
    fuel_type = st.selectbox("Fuel Type", label_encoders["Fuel type"].classes_)
    gear_box = st.selectbox("Gear Box Type", label_encoders["Gear box type"].classes_)

with col2:
    prod_year = st.number_input("Production Year", 1990, 2025, 2015)
    mileage = st.number_input("Mileage (km)", 0, 500000, 50000)
    levy = st.number_input("Levy", 0, 5000, 500)
    airbags = st.slider("Airbags", 0, 16, 4)

st.markdown("")
predict_button = st.button("🚀 Predict Car Price")

# ---------------- PREDICTION ----------------
if predict_button:
    input_data = {
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Prod. year": prod_year,
        "Category": category,
        "Fuel type": fuel_type,
        "Mileage": mileage,
        "Gear box type": gear_box,
        "Airbags": airbags
    }
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in ["Manufacturer", "Category", "Fuel type", "Gear box type"]:
        df[col] = label_encoders[col].transform(df[col])

    df = df[selected_features]
    prediction = model.predict(df)[0]

    st.markdown(
        f'<div class="prediction-card">Estimated Car Price 💰<br>${prediction:,.2f}</div>',
        unsafe_allow_html=True
    )

# ---------------- MODEL INFO ----------------
st.markdown("---")
st.markdown('<p class="section-title">About This AI Model</p>', unsafe_allow_html=True)
st.write("""
This app predicts the **resale price of a car** using a Random Forest ML model.

Key features:
• Advanced feature preprocessing & cleaning  
• Feature selection using statistical methods  
• Real-time price prediction based on car attributes
""")