
import streamlit as st
import pandas as pd
import joblib
import os
import time
from catboost import CatBoostRegressor

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="HousePrice AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Custom CSS — animated, modern UI
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.16), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(14,165,233,0.14), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(168,85,247,0.10), transparent 30%),
            #080b14;
        color: #f8fafc;
    }

    /* Animated background glow */
    .stApp::before {
        content: "";
        position: fixed;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        background: rgba(99,102,241,0.08);
        filter: blur(80px);
        top: 10%;
        left: -120px;
        z-index: -1;
        animation: floatGlow 8s ease-in-out infinite alternate;
    }

    .stApp::after {
        content: "";
        position: fixed;
        width: 380px;
        height: 380px;
        border-radius: 50%;
        background: rgba(14,165,233,0.07);
        filter: blur(90px);
        bottom: 0;
        right: -100px;
        z-index: -1;
        animation: floatGlow2 10s ease-in-out infinite alternate;
    }

    @keyframes floatGlow {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(90px, 60px) scale(1.25); }
    }

    @keyframes floatGlow2 {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(-80px, -50px) scale(1.2); }
    }

    /* Hero */
    .hero {
        padding: 38px 32px;
        margin: 10px 0 28px 0;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        background: linear-gradient(
            135deg,
            rgba(30,41,59,0.80),
            rgba(15,23,42,0.65)
        );
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 70px rgba(0,0,0,0.30);
        animation: heroIn 0.8s ease-out;
    }

    @keyframes heroIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(129,140,248,0.35);
        color: #c7d2fe;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.4px;
        margin-bottom: 14px;
    }

    .hero h1 {
        font-size: clamp(32px, 5vw, 58px);
        line-height: 1.05;
        margin: 0;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #c7d2fe, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: gradientMove 4s linear infinite;
    }

    @keyframes gradientMove {
        to { background-position: 200% center; }
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.7;
        max-width: 850px;
        margin-top: 16px;
    }

    /* Cards */
    .glass-card {
        padding: 22px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(15,23,42,0.62);
        backdrop-filter: blur(16px);
        box-shadow: 0 15px 45px rgba(0,0,0,0.18);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(129,140,248,0.35);
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        margin: 8px 0 4px 0;
        color: #f8fafc;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 20px;
    }

    /* Feature information */
    .feature-info {
        margin: 5px 0 18px 0;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(129,140,248,0.13);
        color: #cbd5e1;
        font-size: 12px;
        line-height: 1.6;
    }

    .feature-info b {
        color: #e0e7ff;
    }

    /* Input styling */
    div[data-testid="stNumberInput"] {
        transition: transform 0.2s ease;
    }

    div[data-testid="stNumberInput"]:hover {
        transform: translateY(-2px);
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 1rem;
        font-size: 16px;
        font-weight: 800;
        color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        background-size: 200% auto;
        box-shadow: 0 10px 30px rgba(99,102,241,0.28);
        transition: all 0.25s ease;
        animation: buttonGradient 4s linear infinite;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 40px rgba(99,102,241,0.40);
    }

    @keyframes buttonGradient {
        to { background-position: 200% center; }
    }

    /* Prediction result */
    .prediction {
        padding: 28px;
        margin-top: 22px;
        border-radius: 24px;
        text-align: center;
        border: 1px solid rgba(34,211,238,0.22);
        background:
            linear-gradient(135deg, rgba(8,47,73,0.75), rgba(30,27,75,0.70));
        box-shadow: 0 15px 55px rgba(6,182,212,0.12);
        animation: resultIn 0.7s ease-out;
    }

    @keyframes resultIn {
        from { opacity: 0; transform: scale(0.94) translateY(15px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }

    .prediction-label {
        color: #a5f3fc;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .prediction-value {
        font-size: clamp(34px, 5vw, 56px);
        font-weight: 800;
        margin-top: 8px;
        color: white;
    }

    /* Footer */
    .footer {
        margin-top: 50px;
        padding: 22px;
        text-align: center;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #94a3b8;
        font-size: 13px;
    }

    .footer strong {
        color: #e2e8f0;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 14px;
        flex-wrap: wrap;
        margin-top: 12px;
    }

    .footer-links a {
        color: #c7d2fe;
        text-decoration: none;
        font-weight: 700;
        padding: 8px 14px;
        border-radius: 999px;
        border: 1px solid rgba(129,140,248,0.35);
        background: rgba(99,102,241,0.12);
        transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
    }

    .footer-links a:hover {
        transform: translateY(-1px);
        background: rgba(99,102,241,0.2);
        border-color: rgba(129,140,248,0.55);
    }

    /* Hide default menu/header/footer for cleaner app */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed.csv")
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "condition",
    "grade",
    "yr_built",
    "lat",
    "long",
]


def train_best_model():
    training_data = pd.read_csv(DATA_PATH)
    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data["price"]

    trained_model = CatBoostRegressor(random_state=42, verbose=0)
    trained_model.fit(X_train, y_train)
    joblib.dump(trained_model, MODEL_PATH)
    return trained_model


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file not found at `{MODEL_PATH}`. "
            "Place `best_model.pkl` inside the `models` folder."
        )

    try:
        loaded_model = joblib.load(MODEL_PATH)
    except Exception as exc:
        return None, f"Could not load the model: {exc}"

    if hasattr(loaded_model, "is_fitted") and not loaded_model.is_fitted():
        try:
            return train_best_model(), None
        except Exception as exc:
            return None, f"Could not train the saved model: {exc}"

    return loaded_model, None


model, model_error = load_model()

# ---------------------------------------------------------
# Hero section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🤖 MACHINE LEARNING • REGRESSION</div>
        <h1>HousePrice AI</h1>
        <p>
            An interactive house-price prediction application powered by a trained
            machine learning pipeline. Enter the property details below and let the
            model estimate the property's market price.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(model_error)

# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown('<div class="section-title">🏠 Property Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Provide the features used by the trained model.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-info">
            <b>Feature guide:</b><br>
            <b>sqft_living</b> = interior living area in square feet •
            <b>sqft_lot</b> = total lot area •
            <b>floors</b> = number of floors •
            <b>condition</b> = overall condition rating •
            <b>grade</b> = construction/design quality rating •
            <b>yr_built</b> = year the house was built •
            <b>lat</b> = latitude •
            <b>long</b> = longitude.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        bedrooms = st.number_input(
            "🛏️ Bedrooms",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=1.0,
            help="Number of bedrooms in the house.",
        )

        bathrooms = st.number_input(
            "🛁 Bathrooms",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            help="Number of bathrooms in the house.",
        )

        sqft_living = st.number_input(
            "📐 Living Area — sqft_living",
            min_value=100.0,
            max_value=20000.0,
            value=1800.0,
            step=50.0,
            help="Interior living area of the house in square feet.",
        )

        sqft_lot = st.number_input(
            "🌳 Lot Area — sqft_lot",
            min_value=300.0,
            max_value=2000000.0,
            value=5000.0,
            step=100.0,
            help="Total land/lot area in square feet.",
        )

        floors = st.number_input(
            "🏢 Floors",
            min_value=1.0,
            max_value=4.0,
            value=1.0,
            step=0.5,
            help="Number of floors in the property.",
        )

        condition = st.slider(
            "🛠️ Condition",
            min_value=1,
            max_value=5,
            value=3,
            help="Overall condition rating. Higher values indicate better condition.",
        )

    with c2:
        grade = st.slider(
            "⭐ Grade",
            min_value=1,
            max_value=13,
            value=7,
            help="Construction/design quality rating used by the dataset.",
        )

        yr_built = st.number_input(
            "📅 Year Built",
            min_value=1800,
            max_value=2026,
            value=2000,
            step=1,
            help="Year in which the house was built.",
        )

        lat = st.number_input(
            "📍 Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=47.5,
            step=0.0001,
            format="%.4f",
            help="Geographic latitude of the property.",
        )

        long = st.number_input(
            "🧭 Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-122.2,
            step=0.0001,
            format="%.4f",
            help="Geographic longitude of the property.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🚀 Predict House Price")

with right:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-size:42px;">🏡</div>
            <h2 style="margin:8px 0;color:#f8fafc;">How it works</h2>
            <p style="color:#94a3b8;line-height:1.7;">
                Your property features are collected and passed to the saved
                <b style="color:#c7d2fe;">best_model.pkl</b> pipeline.
                The pipeline performs the required preprocessing and generates
                the predicted house price.
            </p>
            <hr style="border-color:rgba(255,255,255,0.08);">
            <p style="color:#cbd5e1;line-height:1.8;">
                <b>1.</b> Enter property features<br>
                <b>2.</b> Submit the prediction request<br>
                <b>3.</b> ML pipeline processes the inputs<br>
                <b>4.</b> Receive the estimated price
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_clicked:
        if model is None:
            st.warning("Prediction is unavailable because the model could not be loaded.")
        else:
            input_df = pd.DataFrame(
                {
                    "bedrooms": [bedrooms],
                    "bathrooms": [bathrooms],
                    "sqft_living": [sqft_living],
                    "sqft_lot": [sqft_lot],
                    "floors": [floors],
                    "condition": [condition],
                    "grade": [grade],
                    "yr_built": [yr_built],
                    "lat": [lat],
                    "long": [long],
                }
            )[FEATURE_COLUMNS]

            try:
                with st.spinner("🧠 AI is analyzing the property..."):
                    time.sleep(0.7)
                    prediction = model.predict(input_df)[0]

                st.markdown(
                    f"""
                    <div class="prediction">
                        <div class="prediction-label">Estimated House Price</div>
                        <div class="prediction-value">${prediction:,.0f}</div>
                        <div style="color:#94a3b8;margin-top:8px;">
                            Prediction generated by your trained ML pipeline
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.balloons()

                with st.expander("🔎 View submitted features"):
                    st.dataframe(input_df, use_container_width=True)

            except Exception as exc:
                st.error(
                    "Prediction failed. Make sure the feature names and feature order "
                    "match the model used during training."
                )
                st.exception(exc)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <strong>HousePrice AI</strong> • Machine Learning Regression Project<br><br>
        Built by <strong>Vinay Kumar</strong> • AI & Machine Learning Engineering<br>
        <div class="footer-links">
            <a href="https://github.com/vinaysingh-05" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="https://www.linkedin.com/in/vinay-kumar0805/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        </div>
        Model-powered house price prediction using a saved ML pipeline
    </div>
    """,
    unsafe_allow_html=True,
)
