from flask import Flask, request, jsonify, render_template
from model.rag_modelv10 import ask_question, refresh_web_data
from model.services.weather_service import fetch_current_weather, fetch_forecast, geocode
from model.services.warning_service import fetch_weather_warning
from model.services.phivolcs_earthquake_parser import fetch_and_parse_latest_earthquake
from model.contexts.ph_locations import PH_CITIES_MUNICIPALITIES
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.route("/api/cities")
def get_cities():
    js_ready = {
        city: f"{city}, Philippines"
        for city, meta in PH_CITIES_MUNICIPALITIES.items()
    }
    return jsonify(js_ready)

@app.route('/', endpoint='index')
def index():
    return render_template('index.html', title='DisasterAlertBot')

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"answer": "Error: No data received"}), 400
        message = data.get("message", "").strip()
        if not message:
            logger.warning("Empty message received")
            return jsonify({"answer": "Please enter a question."}), 400
        logger.info(f"Received question: {message}")
        location = data.get("location")
        answer = ask_question(
            message,
            user_location=location if isinstance(location, str) else None,
            session_id=str(data.get("session_id") or "default")
        )
        logger.info(f"Generated answer: {answer[:100]}...")
        return jsonify({"answer": answer}), 200
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({"answer": "Sorry, I encountered an error. Please try again."}), 500

@app.route("/refresh", methods=["POST"])
def refresh():
    # Re-scraping and re-embedding is heavy, so only allow it from this computer
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return jsonify({"status": "error", "message": "Not allowed"}), 403
    try:
        logger.info("Refreshing web data...")
        refresh_web_data()
        logger.info("Web data refreshed successfully")
        return jsonify({"status": "success", "message": "Web sources refreshed!"}), 200
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": "Refresh failed."}), 500

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
    elif condition_id == 801:
        return "bi-cloud-sun"
    elif condition_id == 802:
        return "bi-cloud-sun"
    elif condition_id == 803 or condition_id == 804:
        return "bi-cloud"
    else:
        return "bi-cloud"

@app.route("/weather", methods=["GET"])
def get_weather():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "Please provide a city name via ?city=..."}), 400
    coords = geocode(city, OPENWEATHER_API_KEY)
    if not coords:
        return jsonify({"error": "Unable to geocode city."}), 400
    lat, lon, display_city = coords
    try:
        weather = fetch_current_weather(lat, lon, OPENWEATHER_API_KEY)
        weather.update({"city": city, "display_city": display_city, "provider": "OpenWeather"})
        return jsonify(weather), 200
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return jsonify({"error": "Weather data is unavailable right now."}), 500

@app.route("/forecast", methods=["GET"])
def get_forecast():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "Please provide a city name via ?city=..."}), 400
    coords = geocode(city, OPENWEATHER_API_KEY)
    if not coords:
        return jsonify({"error": "Unable to geocode city."}), 400
    lat, lon, display_city = coords
    try:
        forecast = fetch_forecast(lat, lon, OPENWEATHER_API_KEY)
        forecast["display_city"] = display_city
        return jsonify(forecast), 200
    except Exception as e:
        logger.warning(f"Forecast fetch failed: {e}")
        return jsonify({"error": "Forecast data is unavailable right now."}), 500

@app.route("/weather_warning", methods=["GET"])
def weather_warning():
    result = fetch_weather_warning()
    if not result:
        return jsonify({"error": "No PAGASA bulletin available"}), 500
    return jsonify(result), 200

@app.route("/earthquake_warning", methods=["GET"])
def earthquake_warning():
    earthquake_data = fetch_and_parse_latest_earthquake()

    if not earthquake_data:
        return jsonify({"error": "No significant earthquake data available"}), 500
    
    return jsonify({
        "success": True,
        "data": earthquake_data
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Local only and without the debugger, which would let anyone who can reach it run code
    app.run(host='127.0.0.1', port=5000)