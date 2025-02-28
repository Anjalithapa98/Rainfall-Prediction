import pandas as pd
import joblib

# Manually setting weather data for testing (without changing column order)
weather_data = [
    1200.0,   # Pressure (mb)
    1.0,     # Temperature (°C)
    0,       # Humidity (%)
    0,       # Cloud cover (%)
    7.0,      # Sunshine (UV Index)
    0,      # Wind direction (degrees)
    0.0      # Wind speed (km/h)
]

# Column names in the required order
column_names = ['pressure', 'temparature', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Load the pre-trained model
model = joblib.load("random_forest_model.joblib")

# Create DataFrame using modified weather data
input_df = pd.DataFrame([weather_data], columns=column_names)

# Print the input DataFrame to verify
print("Modified Input DataFrame:\n", input_df)

# Predict probability of rainfall
rainfall_probability = model.predict_proba(input_df)[0][1] * 100  # Probability of rainfall

# Print prediction result as a percentage
print(f"Chance of Rainfall: {rainfall_probability:.2f}%")
