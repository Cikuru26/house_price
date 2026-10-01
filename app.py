from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

MODEL_PATH = Path(__file__).parent / "house_price_model.sav"

# Ranges seen in the cleaned training data (house_price_cleaned.csv)
TRAIN_RANGES = {
    "Area_m2": (26.0, 426.4, "Area (m²)"),
    "Bedrooms": (1, 6, "Bedrooms"),
    "Bathrooms": (1, 5, "Bathrooms"),
    "House_Age_Years": (0.4, 75.0, "House age (years)"),
    "Distance_to_City_km": (0.07, 27.4, "Distance to city centre (km)"),
    "Parking_Spaces": (0, 3, "Parking spaces"),
}
NEIGHBORHOODS = ["Gasabo", "Huye", "Kicukiro", "Kigali City", "Musanze", "Nyarugenge"]
TYPICAL_ERROR = 20  # test-set RMSE of the model, in million RWF (about 19.8)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🏠 House Price Predictor")
st.write(
    "Enter the details of a house to get an instant estimate of its market price "
    "in million Rwandan francs (RWF). The estimate comes from a multiple linear "
    "regression model trained on past house sales."
)

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (m²)", min_value=10.0, max_value=1000.0,
                           value=106.0, step=5.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=15, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=15, value=3, step=1)
    neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS)
with col2:
    age = st.number_input("House age (years)", min_value=0.0, max_value=150.0,
                          value=7.0, step=1.0)
    distance = st.number_input("Distance to city centre (km)", min_value=0.0,
                               max_value=100.0, value=3.3, step=0.5)
    parking = st.number_input("Parking spaces", min_value=0, max_value=10,
                              value=1, step=1)

inputs = {
    "Area_m2": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "House_Age_Years": age,
    "Distance_to_City_km": distance,
    "Parking_Spaces": parking,
}

# Warn when a value is outside the range the model was trained on
out_of_range = [
    f"**{label}** = {inputs[col]:g} (training range: {lo:g} to {hi:g})"
    for col, (lo, hi, label) in TRAIN_RANGES.items()
    if not lo <= inputs[col] <= hi
]
if out_of_range:
    st.warning(
        "Some values are outside the range the model was trained on, so the "
        "estimate may be unreliable:\n\n- " + "\n- ".join(out_of_range)
    )

if st.button("Predict price", type="primary"):
    # One-row DataFrame with the exact column names used in training
    house = pd.DataFrame([{**inputs, "Neighborhood": neighborhood}])
    price = float(model.predict(house)[0])

    if price <= 0:
        st.error(
            "The model returned a non-positive price for these inputs. "
            "Please check the values: they are likely outside the realistic range."
        )
    else:
        st.metric("Estimated price", f"{price:,.1f} million RWF")
        st.caption(
            f"That is about {price * 1_000_000:,.0f} RWF. The model's typical error is "
            f"around ±{TYPICAL_ERROR} million RWF, so a reasonable range is "
            f"{max(price - TYPICAL_ERROR, 0):,.0f} to {price + TYPICAL_ERROR:,.0f} million RWF."
        )

st.divider()
st.caption("Model: multiple linear regression (scikit-learn). Prices in million RWF.")