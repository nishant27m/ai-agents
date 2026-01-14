"""Weather Agent: Suggests office attire based on 7-day weather forecast.

Uses Open-Meteo API (free, no API key required) to fetch weather data.
Returns JSON with weather and AI-powered clothing suggestions.

Usage:
    python3 weather_agent.py              # Default: New York in JSON
    python3 weather_agent.py London       # London in JSON
    python3 weather_agent.py Tokyo        # Tokyo in JSON
    python3 weather_agent.py Singapore markdown  # Singapore in Markdown

Output: JSON (default) or Markdown report
"""
from dotenv import load_dotenv
load_dotenv()

import requests
import json
from datetime import datetime
from langsmith import traceable
from langchain_google_genai import GoogleGenAILLM


@traceable(name="Get Weather Data", run_type="tool")
def get_weather_forecast(location: str = "New York", days: int = 7) -> dict:
    """
    Fetch 7-day weather forecast using Open-Meteo API (free, no key required).
    
    Args:
        location: City name (e.g., "London", "Tokyo", "New York")
        days: Number of days to forecast (default: 7)
    
    Returns:
        Dictionary with daily weather data including temp, rain, wind, and weather code
    """
    try:
        # Geocode location to get latitude/longitude
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        print(f"🌍 Fetching coordinates for: {location}")
        geo_response = requests.get(geo_url, params=geo_params, timeout=10)
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return {"error": f"Location '{location}' not found"}
        
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")
        
        print(f"  Found: {name}, {country} ({lat}, {lon})")
        
        # Fetch weather forecast
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "windspeed_10m_max"
            ],
            "timezone": "auto"
        }
        
        print(f"📡 Fetching weather forecast for {days} days...")
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_data = weather_response.json()
        
        # Parse daily data
        daily_data = weather_data.get("daily", {})
        forecast_days = []
        
        for i in range(min(days, len(daily_data.get("time", [])))):
            day_info = {
                "date": daily_data["time"][i],
                "temp_max": daily_data["temperature_2m_max"][i],
                "temp_min": daily_data["temperature_2m_min"][i],
                "precipitation_mm": daily_data["precipitation_sum"][i],
                "precipitation_prob": daily_data["precipitation_probability_max"][i],
                "wind_speed_kmh": daily_data["windspeed_10m_max"][i],
                "weather_code": daily_data["weather_code"][i],
            }
            day_info["weather_description"] = _interpret_weather_code(day_info["weather_code"])
            forecast_days.append(day_info)
        
        return {
            "location": f"{name}, {country}",
            "forecast_days": forecast_days
        }
    
    except Exception as e:
        return {"error": f"Failed to fetch weather: {str(e)}"}


def _interpret_weather_code(code: int) -> str:
    """Convert WMO weather code to description."""
    code_descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Foggy with rime",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Light showers",
        81: "Moderate showers",
        82: "Heavy showers",
        85: "Light snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
    }
    return code_descriptions.get(code, f"Weather code {code}")


def _get_weather_emoji(description: str) -> str:
    """Get emoji for weather condition."""
    desc_lower = description.lower()
    
    emoji_map = {
        'clear': '☀️',
        'sunny': '☀️',
        'mainly clear': '🌤️',
        'partly cloudy': '⛅',
        'overcast': '☁️',
        'cloudy': '☁️',
        'drizzle': '🌦️',
        'rain': '🌧️',
        'heavy rain': '⛈️',
        'thunderstorm': '⛈️',
        'snow': '❄️',
        'showers': '🌧️',
        'foggy': '🌫️',
        'fog': '🌫️',
    }
    
    for key, emoji in emoji_map.items():
        if key in desc_lower:
            return emoji
    return '🌤️'


@traceable(name="Suggest Office Attire", run_type="tool")
def suggest_office_attire(weather_data: dict) -> str:
    """
    Use LLM to suggest appropriate office wear based on weather forecast.
    
    Args:
        weather_data: Dictionary with location and forecast_days from get_weather_forecast
    
    Returns:
        String with clothing suggestions for each day
    """
    if "error" in weather_data:
        return f"Cannot suggest attire: {weather_data['error']}"
    
    location = weather_data.get("location", "Unknown")
    forecast = weather_data.get("forecast_days", [])
    
    # Format weather data for LLM
    forecast_text = f"\n\nWeather forecast for {location}:\n"
    for day in forecast:
        forecast_text += f"""
Day: {day['date']}
- High: {day['temp_max']}°C, Low: {day['temp_min']}°C
- Condition: {day['weather_description']}
- Precipitation: {day['precipitation_mm']}mm ({day['precipitation_prob']}% chance)
- Wind: {day['wind_speed_kmh']} km/h
"""
    
    # Use LLM to generate suggestions
    llm = GoogleGenAILLM(model="gemini-2.5-flash")
    
    prompt = f"""Based on this weather forecast for an office worker, suggest appropriate office attire for each day.
Consider:
- Temperature comfort (formal vs casual layers)
- Rain/precipitation (waterproof jacket, umbrella needed)
- Wind (consider windbreaker)
- Professional office dress code

Be practical and specific. Format as a numbered list (1., 2., 3., etc), one suggestion per day.
Keep each day's suggestion concise but actionable (2-3 sentences max).
{forecast_text}

Provide specific, actionable clothing recommendations:"""
    
    suggestion = llm(prompt)
    return suggestion




def _parse_daily_attire(attire_suggestion: str, forecast_days: list) -> dict:
    """Parse attire suggestions into day-by-day dictionary."""
    daily_attire = {}
    attire_lines = attire_suggestion.strip().split('\n')
    
    for line in attire_lines:
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            try:
                day_num = int(line.split('.')[0])
                if day_num <= len(forecast_days):
                    date_key = forecast_days[day_num - 1]['date']
                    suggestion = line.split('.', 1)[1].strip() if '.' in line else line
                    daily_attire[date_key] = suggestion
            except:
                pass
    return daily_attire


def build_json_data(location: str, weather_data: dict, attire_suggestion: str) -> dict:
    """Build JSON data structure with weather and attire information."""
    forecast = weather_data.get("forecast_days", [])
    location_name = weather_data.get("location", "Unknown")
    
    # Parse attire suggestions
    daily_attire = _parse_daily_attire(attire_suggestion, forecast)
    
    # Build forecast list
    forecast_list = []
    for i, day in enumerate(forecast):
        date = day['date']
        weather_desc = day['weather_description']
        attire_text = daily_attire.get(date, "Professional business casual attire recommended.")
        
        forecast_list.append({
            "day_number": i + 1,
            "date": date,
            "weather": {
                "emoji": _get_weather_emoji(weather_desc),
                "description": weather_desc,
                "temperature": {
                    "min_celsius": day['temp_min'],
                    "max_celsius": day['temp_max']
                },
                "precipitation": {
                    "amount_mm": day['precipitation_mm'],
                    "probability_percent": day['precipitation_prob']
                },
                "wind_speed_kmh": day['wind_speed_kmh']
            },
            "attire_suggestion": attire_text
        })
    
    # Return complete JSON
    return {
        "metadata": {
            "location": location_name,
            "generated_at": datetime.now().isoformat(),
            "total_days": len(forecast),
            "source": "Weather Agent (Open-Meteo API + Gemini AI)"
        },
        "forecast": forecast_list
    }


@traceable(name="Weather Agent Chain", run_type="chain")
def run_weather_agent(location: str = "New York", days: int = 7, output_format: str = "json") -> dict:
    """
    Main agent that fetches weather and returns data.
    
    Args:
        location: City name (e.g., "London", "Tokyo")
        days: Number of days to forecast
        output_format: "json" (default) or "markdown"
    
    Returns:
        Dictionary with weather and attire data
    """
    print(f"\n{'='*70}")
    print(f"🤖 WEATHER AGENT - 7-Day Office Attire Recommendation")
    print(f"{'='*70}\n")
    
    # Fetch weather
    weather_data = get_weather_forecast(location=location, days=days)
    if "error" in weather_data:
        print(f"❌ Error: {weather_data['error']}")
        return {"error": weather_data['error']}
    
    print(f"✓ Weather data fetched for {weather_data['location']}\n")
    
    # Generate attire suggestions
    print("🧠 Generating attire suggestions with AI...")
    attire_suggestion = suggest_office_attire(weather_data)
    print("✓ Suggestions generated\n")
    
    # Build JSON data
    json_data = build_json_data(location, weather_data, attire_suggestion)
    
    # Output
    if output_format.lower() == "json":
        print("📊 OUTPUT (JSON):")
        print("─" * 70)
        print(json.dumps(json_data, indent=2))
        print("─" * 70)
    
    print(f"{'='*70}\n")
    print(f"✅ Check LangSmith: https://smith.langchain.com/projects/ai-agent\n")
    
    return json_data


if __name__ == "__main__":
    import sys
    
    # Get location from command line or use default
    location = sys.argv[1] if len(sys.argv) > 1 else "New York"
    
    # Get output format (json or markdown)
    output_format = sys.argv[2] if len(sys.argv) > 2 else "json"
    
    run_weather_agent(location=location, days=7, output_format=output_format)
