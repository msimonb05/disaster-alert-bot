import requests
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple, Dict
import logging
from model.contexts.ph_locations import HARDCODED_LOCATIONS

logger = logging.getLogger(__name__)

# Calls OpenWeather without letting the API key from the URL show up in error messages
def openweather_get(url: str, params: Dict):
    try:
        r = requests.get(url, params=params, timeout=8)
    except requests.RequestException as e:
        raise RuntimeError(f"OpenWeather request failed ({type(e).__name__})") from None
    if not r.ok:
        raise RuntimeError(f"OpenWeather request failed with status {r.status_code}")
    return r.json()

# Find a city's coordinates, checking the hardcoded list before OpenWeather and Nominatim
def geocode(city_name: str, api_key: Optional[str]) -> Optional[Tuple[float, float, str]]:
    try:
        return _lookup_city(city_name, api_key)
    except Exception as e:
        logger.warning(f"Geocode failed for {city_name}: {e}")
        return None

# Failed lookups raise, so only the cities that were found get cached
@lru_cache(maxsize=256)
def _lookup_city(city_name: str, api_key: Optional[str]) -> Tuple[float, float, str]:
    clean_name = city_name.replace(", Philippines", "").strip()

    if clean_name in HARDCODED_LOCATIONS:
        print(f"[GEOCODE] Using hardcoded location for: {clean_name}")
        return HARDCODED_LOCATIONS[clean_name]

    if api_key:
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": f"{clean_name},PH", "limit": 1, "appid": api_key}
        try:
            arr = openweather_get(url, params)
        except RuntimeError as e:
            print(f"[GEOCODE] {e}, trying Nominatim instead")
            arr = None
        if isinstance(arr, list) and len(arr) > 0:
            lat = arr[0].get("lat")
            lon = arr[0].get("lon")
            name = arr[0].get("name") or city_name
            state = arr[0].get("state")
            country = arr[0].get("country")
            display = f"{name}" + (f", {state}" if state else "") + (f", {country}" if country else "")
            return float(lat), float(lon), display

    # Search by the plain city name and only inside the Philippines
    nom_url = "https://nominatim.openstreetmap.org/search"
    params = {"q": clean_name, "countrycodes": "ph", "format": "json", "limit": 1}
    arr = requests.get(nom_url, params=params, headers={"User-Agent": "DisasterAlertBot/1.0"}, timeout=6).json()
    if isinstance(arr, list) and len(arr) > 0:
        lat = arr[0].get("lat")
        lon = arr[0].get("lon")
        display = arr[0].get("display_name", city_name)
        return float(lat), float(lon), display

    raise LookupError("no matching place")

def get_weather_icon(condition_main: str, condition_id: int) -> str:
    if condition_id >= 200 and condition_id < 300:
        return "bi-cloud-lightning-rain"
    elif condition_id >= 300 and condition_id < 400:
        return "bi-cloud-drizzle"
    elif condition_id >= 500 and condition_id < 600:
        if condition_id == 500 or condition_id == 520:
            return "bi-cloud-rain"
        else:
            return "bi-cloud-rain-heavy"
    elif condition_id >= 600 and condition_id < 700:
        return "bi-cloud-snow"
    elif condition_id >= 700 and condition_id < 800:
        return "bi-cloud-fog"
    elif condition_id == 800:
        return "bi-brightness-high"
    elif condition_id in [801, 802]:
        return "bi-cloud-sun"
    elif condition_id in [803, 804]:
        return "bi-cloud"
    return "bi-cloud"

def fetch_current_weather(lat: float, lon: float, api_key: str) -> Dict:
    data = openweather_get(
        "https://api.openweathermap.org/data/2.5/weather",
        {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    )

    main = data.get("main", {})
    weather = data.get("weather", [{}])[0]
    wind = data.get("wind", {})

    condition_main = weather.get("main", "Unknown")
    condition_desc = weather.get("description", "").title()
    condition_id = weather.get("id", 800)

    return {
        "temp": round(main.get("temp", 0)),
        "feels_like": round(main.get("feels_like", 0)),
        "humidity": main.get("humidity"),
        "wind": round(wind.get("speed", 0) * 3.6, 1),
        "condition": condition_desc,
        "icon": get_weather_icon(condition_main, condition_id),
        "fetched_at": datetime.utcnow().isoformat() + "Z"
    }

def fetch_forecast(lat: float, lon: float, api_key: str, days: int = 5) -> Dict:
    data = openweather_get(
        "https://api.openweathermap.org/data/2.5/forecast",
        {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "cnt": 40}
    )

    forecast_list = data.get("list", [])
    daily_data: Dict[str, Dict] = {}

    for item in forecast_list:
        dt_txt = item.get("dt_txt", "")
        date = dt_txt.split(" ")[0]

        temp = item.get("main", {}).get("temp")
        temp_min = item.get("main", {}).get("temp_min")
        temp_max = item.get("main", {}).get("temp_max")
        weather = item.get("weather", [{}])[0]
        condition_id = weather.get("id", 800)
        condition_main = weather.get("main", "")

        if date not in daily_data:
            daily_data[date] = {
                "temps": [],
                "mins": [],
                "maxs": [],
                "condition_id": condition_id,
                "condition_main": condition_main
            }

        if temp is not None:
            daily_data[date]["temps"].append(temp)
        if temp_min is not None:
            daily_data[date]["mins"].append(temp_min)
        if temp_max is not None:
            daily_data[date]["maxs"].append(temp_max)

    daily = []
    for date in sorted(daily_data.keys())[:days]:
        day = daily_data[date]
        avg_temp = round(sum(day["temps"]) / len(day["temps"])) if day["temps"] else None
        min_temp = round(min(day["mins"])) if day["mins"] else None
        max_temp = round(max(day["maxs"])) if day["maxs"] else None
        daily.append({
            "date": date,
            "temp": avg_temp,
            "min": min_temp,
            "max": max_temp,
            "condition": day["condition_main"],
            "icon": get_weather_icon(day["condition_main"], day["condition_id"])
        })
    return {"daily": daily}