import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained AI model
model = joblib.load('models/mushroom_model.pkl')

# 2. Web page title and description
st.title("🍄 Mushroom Yield Forecasting System")
st.write("Adjust the environmental factors below to see the predicted mushroom yield.")

st.divider()

# 3. Create interactive sliders for user inputs
st.subheader("🎛️ Polyhouse Sensor Controls")
temp = st.slider("Temperature (°C)", min_value=15.0, max_value=30.0, value=22.0, step=0.1)
humidity = st.slider("Humidity (%)", min_value=70.0, max_value=100.0, value=90.0, step=0.1)
co2 = st.slider("CO2 Levels (ppm)", min_value=500, max_value=1500, value=1000, step=10)

# 4. Make prediction when sliders move
input_data = pd.DataFrame([{
    'Temperature': temp,
    'Humidity': humidity,
    'CO2': co2
}])

predicted_yield = model.predict(input_data)[0]

st.divider()

# 5. Display the prediction beautifully
st.subheader("📊 Forecasted Result")
st.metric(label="Estimated Mushroom Yield", value=f"{predicted_yield:.2f} kg/m²")