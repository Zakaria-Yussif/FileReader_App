import re
import unicodedata
import requests
from datetime import datetime
from django.conf import settings



def extract_city_from_input(user_input):
    match = re.search(r"\b(?:in|for)\s+(?P<city>[a-zA-Z\s]+)", user_input, re.IGNORECASE)
    if match:
        return match.group("city").strip()
    return None


def clean_city_name(city):
    city = unicodedata.normalize('NFKD', city).encode('ascii', 'ignore').decode('utf-8')
    return city.title().strip()


def get_location():
    """Get city from IP address (fallback if no city in input)"""
    url = f'https://ipinfo.io?token={location_key}'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('city')
    except Exception as e:
        print(f"🌐 Location fetch failed: {e}")
        return None


def get_weather(city, units='metric'):
    """Get weather data from OpenWeatherMap API"""
    if not city:
        return {"error": "No city provided"}

    city = clean_city_name(city)

    params = {
        'q': city,
        'appid': weather_key,
        'units': units
    }

    try:
        print(f"🔍 Fetching weather for: {city}")
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        weather = data['weather'][0]
        main = data['main']
        timestamp = data['dt']

        return {
            'city': city,
            'weather': weather['description'],
            'weather_icon': f"http://openweathermap.org/img/wn/{weather['icon']}@2x.png",
            'conditions': weather['main'],
            'day': datetime.fromtimestamp(timestamp).strftime('%A'),
            'time': datetime.fromtimestamp(timestamp).strftime('%H:%M'),
            'weather_temperature': f"{main['temp']}°C",
        }

    except requests.exceptions.HTTPError as e:
        return {"error": f"❌ HTTP error: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"🔌 Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"⚠️ Unexpected error: {str(e)}"}

location_key1 = settings.LOCATION_KEY
location_key = "4b475a9d852574"
weather_key1 = settings.WEATHER_API_KEY
weather_key = "d74ba73a5cb13931ea95f29deccee549"
base_url = 'http://api.openweathermap.org/data/2.5/weather'


def extract_city_from_input(user_input):
    match = re.search(r"\b(?:in|for)\s+(?P<city>[a-zA-Z\s]+)", user_input, re.IGNORECASE)
    if match:
        return match.group("city").strip()
    return None


def clean_city_name(city):
    city = unicodedata.normalize('NFKD', city).encode('ascii', 'ignore').decode('utf-8')
    return city.title().strip()


def get_location():
    """Get city from IP address (fallback if no city in input)"""
    url = f'https://ipinfo.io?token={location_key}'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('city')
    except Exception as e:
        print(f"🌐 Location fetch failed: {e}")
        return None


def get_weather(city, units='metric'):
    """Get weather data from OpenWeatherMap API"""
    if not city:
        return {"error": "No city provided"}

    city = clean_city_name(city)

    params = {
        'q': city,
        'appid': weather_key,
        'units': units
    }

    try:
        print(f"🔍 Fetching weather for: {city}")
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        weather = data['weather'][0]
        main = data['main']
        timestamp = data['dt']

        return {
            'city': city,
            'weather': weather['description'],
            'weather_icon': f"http://openweathermap.org/img/wn/{weather['icon']}@2x.png",
            'conditions': weather['main'],
            'day': datetime.fromtimestamp(timestamp).strftime('%A'),
            'time': datetime.fromtimestamp(timestamp).strftime('%H:%M'),
            'weather_temperature': f"{main['temp']}°C",
        }

    except requests.exceptions.HTTPError as e:
        return {"error": f"❌ HTTP error: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"🔌 Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"⚠️ Unexpected error: {str(e)}"}