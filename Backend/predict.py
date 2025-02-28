import getTodayWeather
import pandas as pd
import joblib

# Fetch weather data
weather_data, column_names = getTodayWeather.get_weather_data()

# Print the fetched weather data and column names
print("Weather Data:", weather_data)
print("Column Names:", column_names)

# Load the pre-trained model
model = joblib.load("random_forest_model.joblib")

# Print the type of the model to confirm it is a model object
print("Model Type:", type(model))

# Create DataFrame using weather data
input_df = pd.DataFrame([weather_data], columns=column_names)

# Print the input DataFrame to verify
print("Input DataFrame:\n", input_df)

# Predict probability of rainfall
rainfall_probability = model.predict_proba(input_df)[0][1] * 100  # Convert to percentage

# Print the rainfall probability
print(f"Chance of Rainfall: {rainfall_probability:.2f}%")

def probability():
    return rainfall_probability
