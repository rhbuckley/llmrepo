import requests
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel
from llmrepo.tools import BaseTool, BaseToolbox, ToolParameter

class CurrentWeather(BaseModel):
    temperature_2m: float
    wind_speed_10m: float
    time: str

class HourlyWeather(BaseModel):
    time: List[str]
    temperature_2m: List[float]
    relative_humidity_2m: List[float]
    wind_speed_10m: List[float]

class WeatherResponse(BaseModel):
    latitude: float
    longitude: float
    timezone: str
    timezone_abbreviation: str
    elevation: float
    current: CurrentWeather
    hourly: HourlyWeather


class WeatherTool(BaseTool):
    """Tool to get weather forecast using Open-Meteo API."""
    name: str = "get_weather"
    description: str = "Get current and hourly weather forecast using coordinates"
    endpoint: str = "https://api.open-meteo.com/v1/forecast"
    parameters: Dict[str, ToolParameter] = {
        "lat": ToolParameter(
            name="lat",
            type="float",
            description="Latitude (-90 to 90)",
            required=True
        ),
        "lon": ToolParameter(
            name="lon",
            type="float",
            description="Longitude (-180 to 180)",
            required=True
        )
    }
    
    def invoke(self, lat: float, lon: float) -> WeatherResponse:
        # Store coordinates in shared context
        self.context.set('last_lat', lat, force_shared=True)
        self.context.set('last_lon', lon, force_shared=True)
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,wind_speed_10m',
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m',
            'timezone': 'auto'
        }
        
        result = requests.get(self.endpoint, params=params)
        result.raise_for_status()
        return WeatherResponse(**result.json())

class WeatherToolbox(BaseToolbox):
    """Toolbox for weather operations using Open-Meteo API."""
    weather = WeatherTool()

if __name__ == "__main__":
    try:
        weather_toolbox = WeatherToolbox()
        
        # Example coordinates for London
        london_lat = 51.5074
        london_lon = -0.1278
        
        # Get weather data
        response = weather_toolbox.weather.invoke(
            lat=london_lat,
            lon=london_lon
        )
        
        # Print current weather
        current = response.current
        print(f"\nCurrent weather at ({london_lat}, {london_lon}):")
        print(f"Time: {current.time}")
        print(f"Temperature: {current.temperature_2m}°C")
        print(f"Wind Speed: {current.wind_speed_10m} m/s")
        
        # Print hourly forecast for the next 24 hours
        print("\nHourly Forecast (next 24 hours):")
        for i in range(24):
            time = datetime.fromisoformat(response.hourly.time[i])
            print(f"\n{time.strftime('%I:%M %p')}:")
            print(f"Temperature: {response.hourly.temperature_2m[i]}°C")
            print(f"Humidity: {response.hourly.relative_humidity_2m[i]}%")
            print(f"Wind Speed: {response.hourly.wind_speed_10m[i]} m/s")
            
    except ValueError as e:
        print(f"\nError: {str(e)}")
    except requests.exceptions.RequestException as e:
        print(f"\nAPI Error: {str(e)}")
