import requests

# ✅ Your actual API key from OpenWeatherMap
OPENWEATHER_API_KEY = "3fde8c7d0e8b7862cc128302d780d66e"

def predict_weather(location: str) -> str:
    url = (
        f"http://api.openweathermap.org/data/2.5/weather?"
        f"q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return "Sorry, could not retrieve weather data for that location."

    data = response.json()
    weather = data["weather"][0]["description"].capitalize()
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]

    return f"{location.title()}: {weather}, {temp}°C (feels like {feels_like}°C)"
