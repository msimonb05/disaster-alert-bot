import re

CURRENT_INFO_KEYWORDS = [
    "current", "now", "today", "latest", "recent", "active", "present",
    "kasalukuyan", "ngayon", "bagong", "kamakailan"
]

LOCATION_KEYWORDS = [
    "here", "near me", "dito", "malapit", "sa amin", "sa lugar ko",
    "my area", "this area", "sa akin", "around me", "nearby",
    "sa bahay", "sa atin", "local", "lokal", "philippines",
    "pilipinas", "ph"
]

WEATHER_KEYWORDS = [
    "weather", "temperature", "panahon", "init", "lamig", "ulan", "rain",
    "sunny", "cloudy", "forecast", "humidity", "wind", "hangin",
    "mainit", "malamig", "umuulan", "maulan", "umbrella", "payong"
]

TYPHOON_KEYWORDS = [
    "typhoon", "bagyo", "tropical cyclone", "tropical depression", "storm",
    "wind signal", "signal no", "signal number", "under signal", "storm signal",
    "tcws", "pagasa bulletin", "storm surge", "cyclone"
]

EARTHQUAKE_KEYWORDS = [
    "earthquake", "lindol", "lumindol", "lumilindol", "tremor", "aftershock",
    "magnitude", "richter", "phivolcs", "seismic", "quake", "epicenter"
]

WEATHER_DATA_KEYWORDS = [
    "temperature", "degrees", "celsius", "fahrenheit", "humidity", "wind speed",
    "forecast", "temperatura", "grados"
]

WEATHER_CONTEXT_KEYWORDS = [
    "dangerous", "safe", "wear", "bring", "umbrella", "should i", "can i",
    "delikado", "ligtas", "pwede", "dapat", "jacket", "payong"
]

TYPHOON_DATA_KEYWORDS = [
    "signal", "signal number", "wind speed", "pressure", "classification",
    "category", "location", "coordinates", "lakas", "kategorya"
]

TYPHOON_CONTEXT_KEYWORDS = [
    "affected", "damage", "casualties", "evacuation", "impact", "news",
    "situation", "update", "naapektuhan", "pinsala", "sitwasyon", "balita"
]

EARTHQUAKE_DATA_KEYWORDS = [
    "magnitude", "depth", "epicenter", "coordinates", "richter", "intensity",
    "lakas", "lalim", "sentro"
]

EARTHQUAKE_CONTEXT_KEYWORDS = [
    "damage", "casualties", "affected", "aftershock", "tsunami", "injuries",
    "safety", "pinsala", "nasaktan", "kaligtasan", "pangamba"
]

GENERAL_CONTEXT_KEYWORDS = [
    "situation", "what's happening", "status", "news", "report", "update",
    "alert", "sitwasyon", "nangyayari", "balita", "ulat", "babala",
    "what happened", "how are"
]

DIRECT_QUESTION_PATTERNS = [
    "what is", "what's the", "ano ang", "ilan ang", "how much", "how many"
]

CONTEXT_QUESTION_PATTERNS = [
    "how", "why", "should i", "can i", "is it", "bakit", "paano", "pwede ba"
]

HYBRID_QUESTION_PATTERNS = [
    "should i", "can i", "is it safe", "pwede ba", "ligtas ba", "okay ba"
]

FOLLOW_UP_CUES = [
    "what about", "how about", "and", "also", "then", "it", "that", "this",
    "there", "tomorrow", "naman", "bukas", "doon", "diyan", "iyan", "yan", "ito"
]

# Match whole words so "rain" won't match "train", but still allow simple
# endings like "typhoons", "raining" or "bagyong"
def keyword_in_text(keyword: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}(?:s|es|d|ed|ing|y|g|ng)?\b", text) is not None

def has_any(text: str, keywords: list) -> bool:
    text = text.lower()
    return any(keyword_in_text(keyword, text) for keyword in keywords)

def count_matches(text: str, keywords: list) -> int:
    text = text.lower()
    return sum(1 for keyword in keywords if keyword_in_text(keyword, text))
