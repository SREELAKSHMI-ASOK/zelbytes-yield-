import joblib
import pandas as pd

# 1. Load the trained AI brain
model = joblib.load('models/mushroom_model.pkl')

# 2. Provide brand new sensor data (Temp, Humidity, CO2)
new_sensor_reading = pd.DataFrame([{
    'Temperature': 22.5,
    'Humidity': 90.0,
    'CO2': 1050
}])

# 3. Let the AI predict the yield!
prediction = model.predict(new_sensor_reading)

print("\n--- AI Prediction Result ---")
print(f"For Temp: 22.5°C, Humidity: 90%, CO2: 1050ppm")
print(f"Predicted Mushroom Yield: {prediction[0]:.2f} kg/m²\n")