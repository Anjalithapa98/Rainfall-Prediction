import requests
from datetime import datetime

# WeatherAPI key and URL
api_key = 'b1a622eef85f4710ad3201823252402'
city = 'kathmandu'  # Replace with the desired city
url = f'http://api.weatherapi.com/v1/current.json?q={city}&key={api_key}'

def get_weather_data():
    try:
        # Send a request to the API
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful (status code 200)
        
        data = response.json()

        # Extract relevant data from the response
        if 'current' in data:
            pressure = data['current']['pressure_mb']  # Pressure in mb
            temperature = data['current']['temp_c']  # Temperature in Celsius
            humidity = data['current']['humidity']  # Humidity in percentage
            cloud = data['current']['cloud']  # Cloudiness in percentage
            windspeed = data['current']['wind_kph']  # Wind speed in km/h
            winddirection = data['current']['wind_degree']  # Wind direction in degrees
            sunshine = data['current']['uv']  # UV index (if available)

            # Return the data as a tuple and column names as a list
            column_names = ['pressure', "temparature", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
            return (pressure, temperature, humidity, cloud, sunshine, winddirection, windspeed), column_names
        else:
            print("Error: The API response does not contain the expected data.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error with the API request: {e}")
        return None
    except KeyError as e:
        print(f"Missing expected data in the API response: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Call the function and store the returned values
weather_data, column_names = get_weather_data() if get_weather_data() else (None, None)

# Return the weather data and column names
def return_weather_data():
    return weather_data, column_names
