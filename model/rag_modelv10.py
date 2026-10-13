import os
from .services.data_scrape import fetch_disaster_data
from .services.serper_service import retrieve_realtime_docs
from .services.weather_service import fetch_current_weather, fetch_forecast, geocode
from .services.warning_service import fetch_weather_warning
from .services.phivolcs_earthquake_parser import fetch_and_parse_latest_earthquake
from .contexts.context_keywords import LOCATION_KEYWORDS, WEATHER_KEYWORDS, TYPHOON_KEYWORDS, EARTHQUAKE_KEYWORDS, \
    WEATHER_DATA_KEYWORDS, WEATHER_CONTEXT_KEYWORDS, TYPHOON_DATA_KEYWORDS, TYPHOON_CONTEXT_KEYWORDS, \
    EARTHQUAKE_DATA_KEYWORDS, EARTHQUAKE_CONTEXT_KEYWORDS, GENERAL_CONTEXT_KEYWORDS, HYBRID_QUESTION_PATTERNS, \
    FOLLOW_UP_CUES, has_any, count_matches
from .contexts.prompt_template import PROMPT_TEMPLATE_V10
from collections import deque
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from typing import List, Optional, Tuple

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = "DisasteAlertBot_Data_v1.pdf"
PDF_PATH = os.path.join(BASE_DIR, "pdf", PDF_FILE)
DB_DIR = os.path.join(BASE_DIR, "chroma_db", "v10")
os.makedirs(DB_DIR, exist_ok=True)
PDF_TOP_K = 3
WEB_TOP_K = 2
DEFAULT_LOCATION = (14.5995, 120.9842, "Manila, Philippines")
PH_TIMEZONE = timezone(timedelta(hours=8))
MAX_FOLLOW_UP_WORDS = 6

""" QUERY INTENT CLASSIFICATION """
class QueryIntent:
    DIRECT_DATA = "direct_data"
    CONTEXT = "context"
    HYBRID = "hybrid"
    GENERAL = "general"

# Data and context keywords for each disaster type
TOPIC_KEYWORDS = {
    "weather": (WEATHER_DATA_KEYWORDS, WEATHER_CONTEXT_KEYWORDS),
    "typhoon": (TYPHOON_DATA_KEYWORDS, TYPHOON_CONTEXT_KEYWORDS),
    "earthquake": (EARTHQUAKE_DATA_KEYWORDS, EARTHQUAKE_CONTEXT_KEYWORDS),
}

# Check if question mentions a specific disaster type
def has_disaster_type(question: str) -> Tuple[bool, Optional[str]]:
    if has_any(question, TYPHOON_KEYWORDS):
        return True, "typhoon"
    if has_any(question, EARTHQUAKE_KEYWORDS):
        return True, "earthquake"
    if has_any(question, WEATHER_KEYWORDS):
        return True, "weather"
    return False, None

# Detect if question is very broad/vague
def is_vague_question(question: str) -> bool:
    return len(question.split()) <= 4 or has_any(question, ["any", "anything", "what's up", "ano meron", "update"])

# Broad questions like "any updates?" or "what's happening?"
def wants_overview(question: str) -> bool:
    return is_vague_question(question) or has_any(question, GENERAL_CONTEXT_KEYWORDS)

# classify the user's query intent to determine retrieval strategy
def classify_query_intent(question: str) -> str:
    print(f"[INTENT CLASSIFIER] Analyzing: {question}")

    has_disaster, disaster_type = has_disaster_type(question)

    data_count = 0
    context_count = 0
    if disaster_type:
        data_keywords, context_keywords = TOPIC_KEYWORDS[disaster_type]
        data_count = count_matches(question, data_keywords)
        context_count = count_matches(question, context_keywords)
    context_count += count_matches(question, GENERAL_CONTEXT_KEYWORDS)

    has_hybrid_pattern = has_any(question, HYBRID_QUESTION_PATTERNS)

    print(f"[INTENT CLASSIFIER] Disaster type: {disaster_type}, Data keywords: {data_count}, Context keywords: {context_count}, Hybrid pattern: {has_hybrid_pattern}")

    if not has_disaster and wants_overview(question):
        intent = QueryIntent.GENERAL
    # Direct data questions ask for specific numbers
    elif data_count >= 1 and context_count == 0 and not has_hybrid_pattern:
        intent = QueryIntent.DIRECT_DATA
    # Context questions ask about impacts, news or the situation
    elif context_count >= 1 and data_count == 0:
        intent = QueryIntent.CONTEXT
    # Hybrid questions need both data and context
    elif (data_count > 0 and context_count > 0) or has_hybrid_pattern:
        intent = QueryIntent.HYBRID
    else:
        intent = QueryIntent.GENERAL

    print(f"[INTENT CLASSIFIER] Result: {intent}")
    return intent

""" CONTEXT REINFORCEMENT """
# Check if weather data is needed
def needs_weather_data(question: str) -> bool:
    if has_any(question, WEATHER_KEYWORDS):
        return True
    return has_any(question, LOCATION_KEYWORDS) and has_any(question, WEATHER_DATA_KEYWORDS + WEATHER_CONTEXT_KEYWORDS)

# Check if typhoon data is needed
def needs_typhoon_data(question: str) -> bool:
    return has_any(question, TYPHOON_KEYWORDS)

# Check if earthquake data is needed
def needs_earthquake_data(question: str) -> bool:
    return has_any(question, EARTHQUAKE_KEYWORDS)

""" SERVICE DATA FETCHING """
# How many days old a PAGASA bulletin is, or None if the date can't be read
def bulletin_age_days(issued: Optional[str]) -> Optional[int]:
    try:
        issued_at = datetime.strptime(issued, "%I:%M %p, %d %B %Y").replace(tzinfo=PH_TIMEZONE)
    except (TypeError, ValueError):
        return None
    return (datetime.now(PH_TIMEZONE) - issued_at).days

# Fetch weather service data
def fetch_weather_service_data(user_location: Optional[str]) -> Optional[Document]:
    try:
        if not OPENWEATHER_API_KEY:
            print("[WEATHER SERVICE] OpenWeather API key not configured")
            return None

        coords = geocode(user_location, OPENWEATHER_API_KEY) if user_location else None
        if not coords:
            print("[WEATHER SERVICE] No valid location, using Manila as default")
            coords = DEFAULT_LOCATION

        lat, lon, display_city = coords

        print(f"[WEATHER SERVICE] Fetching current weather for {display_city}")
        current = fetch_current_weather(lat, lon, OPENWEATHER_API_KEY)

        print(f"[WEATHER SERVICE] Fetching forecast for {display_city}")
        forecast = fetch_forecast(lat, lon, OPENWEATHER_API_KEY)

        weather_text = f"Current Weather in {display_city}:\n"
        weather_text += f"Temperature: {current.get('temp')}°C\n"
        weather_text += f"Feels Like: {current.get('feels_like')}°C\n"
        weather_text += f"Condition: {current.get('condition')}\n"
        weather_text += f"Humidity: {current.get('humidity')}%\n"
        weather_text += f"Wind Speed: {current.get('wind')} km/h\n\n"

        weather_text += f"5-Day Forecast for {display_city}:\n"
        for day in forecast.get('daily', []):
            weather_text += f"- {day.get('date')}: High {day.get('max')}°C, Low {day.get('min')}°C, {day.get('condition')}\n"

        return Document(
            page_content=weather_text,
            metadata={
                "source": f"OpenWeather API - {display_city}",
                "source_type": "weather_service",
                "priority": "highest"
            }
        )
    except Exception as e:
        print(f"[WEATHER SERVICE] Error: {e}")
        return None

# Fetch typhoon service data
def fetch_typhoon_service_data() -> Optional[Document]:
    try:
        print("[TYPHOON SERVICE] Fetching PAGASA bulletin...")
        data = fetch_weather_warning()

        if not data:
            print("[TYPHOON SERVICE] No bulletin retrieved.")
            return None

        typhoon_text = "PAGASA Tropical Cyclone Bulletin\n\n"

        # PAGASA keeps the last bulletin up after a storm is gone, so say when it's old
        age = bulletin_age_days(data.get("issued"))
        if age is not None and age >= 1:
            typhoon_text += (
                f"This is the latest PAGASA bulletin, but it was issued {age} day(s) ago. "
                "This storm may no longer be active and there may be no tropical cyclone right now.\n\n"
            )

        # Basic metadata
        typhoon_text += f"Name: {data.get('name', 'N/A')}\n"
        typhoon_text += f"Classification: {data.get('classification', 'N/A')}\n"
        typhoon_text += f"Issued: {data.get('issued') or 'N/A'}\n\n"

        # Intensity
        intensity = data.get("intensity") or {}
        typhoon_text += "Intensity:\n"
        typhoon_text += f"- Maximum Winds: {intensity.get('max_winds_kmh', 'N/A')} km/h\n"
        typhoon_text += f"- Gustiness: {intensity.get('gustiness_kmh', 'N/A')} km/h\n"
        typhoon_text += f"- Pressure: {intensity.get('pressure_hpa', 'N/A')} hPa\n\n"

        # Movement
        movement = data.get("present_movement") or {}
        direction = movement.get("direction") or "Unavailable"
        speed = movement.get("speed_kmh") or "Unavailable"
        typhoon_text += f"Movement: {direction} at {speed} km/h\n\n"

        # Location
        location = data.get("location") or {}
        typhoon_text += "Location:\n"
        typhoon_text += f"- Description: {location.get('description', 'N/A')}\n"
        typhoon_text += f"- As of: {location.get('as_of', 'N/A')}\n\n"

        # Every wind signal level, highest first
        signals = sorted(data.get("tcws") or [], key=lambda s: s.get("signal_no") or 0, reverse=True)
        if signals:
            typhoon_text += "Tropical Cyclone Wind Signals (TCWS) in effect:\n"
            for signal in signals:
                typhoon_text += f"- Signal No. {signal.get('signal_no')} ({signal.get('impact')}): {signal.get('affected_areas') or 'N/A'}\n"
            typhoon_text += "\n"
        else:
            typhoon_text += "No Tropical Cyclone Wind Signals in effect.\n\n"

        # Source link
        typhoon_text += f"Source: {data.get('source','N/A')}\n"

        return Document(
            page_content=typhoon_text,
            metadata={
                "source": "PAGASA Tropical Cyclone Bulletin",
                "source_type": "typhoon_service",
                "priority": "highest"
            }
        )

    except Exception as e:
        print(f"[TYPHOON SERVICE] Error: {e}")
        return None

# Fetch earthquake service data
def fetch_earthquake_service_data() -> Optional[Document]:
    try:
        print("[EARTHQUAKE SERVICE] Fetching PHIVOLCS data...")
        eq = fetch_and_parse_latest_earthquake()

        if not eq or not eq.get("magnitude"):
            print("[EARTHQUAKE SERVICE] No significant earthquake data")
            return None

        date_time = f"{eq.get('date','N/A')} {eq.get('time','')}".strip()

        # Latest significant earthquake information
        eq_text = "Latest Significant Earthquake (PHIVOLCS):\n\n"
        eq_text += f"Magnitude: {eq.get('magnitude','N/A')}\n"
        eq_text += f"Date & Time: {date_time}\n"
        eq_text += f"Issued On: {eq.get('issued_on','N/A')}\n"
        eq_text += f"Location: {eq.get('location','N/A')}\n"
        eq_text += f"Depth: {eq.get('depth_of_focus','N/A')}\n\n"

        coords = eq.get("coordinates", {})
        if coords:
            eq_text += f"Coordinates: {coords.get('latitude','N/A')}°N, {coords.get('longitude','N/A')}°E\n"

        if eq.get("reported_intensities"):
            eq_text += "\nReported Intensities:\n"
            for item in eq["reported_intensities"]:
                eq_text += f"- Intensity {item['intensity']}: {item['places']}\n"

        if eq.get("aftershock") is not None:
            eq_text += f"\nAftershock Expected: {'Yes' if eq['aftershock'] else 'No'}\n"

        return Document(
            page_content=eq_text,
            metadata={
                "source": "PHIVOLCS Latest Earthquake Bulletin",
                "source_type": "earthquake_service",
                "priority": "highest"
            }
        )

    except Exception as e:
        print(f"[EARTHQUAKE SERVICE] Error: {e}")
        return None

""" VECTOR STORE INITIALIZATION """
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "?", "!"]
)

# Loader for PDF data
def load_pdf_chunks() -> List[Document]:
    try:
        pdf_docs = PyMuPDFLoader(str(PDF_PATH)).load()
        pdf_chunks = TEXT_SPLITTER.split_documents(pdf_docs)
        for chunk in pdf_chunks:
            chunk.metadata["source_type"] = "pdf"
            chunk.metadata["priority"] = "high"
        return pdf_chunks
    except Exception as e:
        print(f"[ERROR] Failed to load PDF: {e}")
        return []

# Loader for web data
def load_web_chunks() -> List[Document]:
    try:
        web_chunks = TEXT_SPLITTER.split_documents(fetch_disaster_data())
        for chunk in web_chunks:
            chunk.metadata["source_type"] = "web"
            chunk.metadata["priority"] = "medium"
        return web_chunks
    except Exception as e:
        print(f"[ERROR] Failed to load web sources: {e}")
        return []

# Open a saved store and only build it when it's still empty, so restarts don't add duplicates
def open_vector_store(name: str, load_chunks) -> Chroma:
    store = Chroma(
        embedding_function=embeddings,
        persist_directory=os.path.join(DB_DIR, name),
        collection_configuration={"hnsw": {"space": "cosine"}}
    )
    if not store.get(limit=1)["ids"]:
        chunks = load_chunks()
        if chunks:
            print(f"[VECTOR STORE] Building {name} with {len(chunks)} chunks")
            store.add_documents(chunks)
    return store

""" ENHANCED RAG MODEL LOGIC WITH INTENT-BASED ROUTING """
# Multilingual model so Filipino and Taglish questions match the English guides
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"prompt": "passage: "},
    query_encode_kwargs={"prompt": "query: "}
)
pdf_vector_store = open_vector_store("pdf_store", load_pdf_chunks)
web_vector_store = open_vector_store("web_store", load_web_chunks)
llm = OllamaLLM(model="llama3.2:3b", temperature=0.2, num_ctx=8192)

# Search the PDF handbook and the scraped web guides
def kb_docs(question: str) -> List[Document]:
    return pdf_vector_store.similarity_search(question, k=PDF_TOP_K) + web_vector_store.similarity_search(question, k=WEB_TOP_K)

def retrieve_data(question: str, user_location: Optional[str] = None) -> Tuple[List[Document], str]:
    print(f"[RETRIEVAL] Analyzing question: {question}")
    if user_location:
        print(f"[RETRIEVAL] User location: {user_location}")

    intent = classify_query_intent(question)

    # Broad questions with no disaster topic get data from every service
    overview = wants_overview(question) and not has_disaster_type(question)[0]
    if overview:
        print("[RETRIEVAL] Broad question - fetching every service")

    # Collect relevant service documents
    service_docs = []

    if overview or needs_weather_data(question):
        print("[RETRIEVAL] Weather query detected - fetching weather service data")
        service_docs.append(fetch_weather_service_data(user_location))

    if overview or needs_typhoon_data(question):
        print("[RETRIEVAL] Typhoon query detected - fetching PAGASA bulletin")
        service_docs.append(fetch_typhoon_service_data())

    if overview or needs_earthquake_data(question):
        print("[RETRIEVAL] Earthquake query detected - fetching PHIVOLCS data")
        service_docs.append(fetch_earthquake_service_data())

    service_docs = [doc for doc in service_docs if doc]

    # Direct data questions are answered by the services, everything else also gets a web search
    serper_docs = []
    if intent != QueryIntent.DIRECT_DATA:
        serper_docs = retrieve_realtime_docs(question, user_location=user_location) or []
        for d in serper_docs:
            d.metadata["source_type"] = "serper"
            d.metadata["priority"] = "medium"

    # Official data first, then search results, then the guides
    knowledge_docs = kb_docs(question)
    print(f"[RETRIEVAL] Retrieved: {len(service_docs)} service + {len(serper_docs)} serper + {len(knowledge_docs)} kb")
    return service_docs + serper_docs + knowledge_docs, intent

# Sessions live in memory, so they reset when the server restarts
user_sessions = {}
MAX_TOPIC_HISTORY = 3

def detect_topic(question: str) -> str:
    return has_disaster_type(question)[1] or "general"

def get_session(session_id: str):
    if session_id not in user_sessions:
        user_sessions[session_id] = {"topic": None, "history": deque(maxlen=MAX_TOPIC_HISTORY)}
    return user_sessions[session_id]

SOURCE_LABELS = {
    "weather_service": "Weather Service (Real-time)",
    "typhoon_service": "PAGASA Bulletin (Real-time)",
    "earthquake_service": "PHIVOLCS Data (Real-time)",
    "pdf": "PDF Handbook",
    "web": "Web Official Source",
    "serper": "Real-time Search"
}

def ask_question(question: str, user_location: Optional[str] = None, session_id: str = "default") -> str:
    print(f"\n[QUERY] User asked: {question}")
    if user_location:
        print(f"[QUERY] User location: {user_location}")

    session = get_session(session_id)
    new_topic = detect_topic(question)
    old_topic = session["topic"]
    retrieval_query = question

    print(f"[TOPIC] Previous topic: {old_topic}, New topic: {new_topic}")

    # Short follow-ups like "what about tomorrow?" stay on the previous topic
    is_follow_up = (
        new_topic == "general"
        and old_topic not in (None, "general")
        and session["history"]
        and len(question.split()) <= MAX_FOLLOW_UP_WORDS
        and has_any(question, FOLLOW_UP_CUES)
    )

    if is_follow_up:
        new_topic = old_topic
        anchor = next((q for q, _ in reversed(session["history"]) if detect_topic(q) != "general"), "")
        retrieval_query = f"{anchor} {question}".strip()
        print(f"[TOPIC] Follow-up question, searching with: {retrieval_query}")
    elif old_topic is not None and new_topic != old_topic:
        print("[TOPIC] Topic changed — clearing chat history.")
        session["history"].clear()

    session["topic"] = new_topic

    history_text = "".join(f"User: {q}\nBot: {a}\n" for q, a in session["history"])
    print(f"[CHAT HISTORY] Using {len(session['history'])} past exchanges")

    docs, intent = retrieve_data(retrieval_query, user_location=user_location)
    print(f"[CONTEXT] Building context from {len(docs)} documents (intent: {intent})")

    context_parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "Unknown")
        # Show the handbook's file name and page instead of the full path on this computer
        if d.metadata.get("source_type") == "pdf":
            source = f"{PDF_FILE}, page {d.metadata.get('page', 0) + 1}"
        label_prefix = SOURCE_LABELS.get(d.metadata.get("source_type"), "Unknown")
        context_parts.append(
            f"[Source {i}: {label_prefix} - {source} | priority={d.metadata.get('priority','')}]"
            f"\n{d.page_content.strip()}"
        )

    context = "\n\n".join(context_parts) or "No relevant context found."
    now = datetime.now(PH_TIMEZONE).strftime("%A, %B %d, %Y %I:%M %p")

    prompt = PROMPT_TEMPLATE_V10.format(
        now=now,
        chat_history=history_text or "None",
        context=context,
        question=question
    )

    print(f"[GENERATING] Calling LLM...")
    answer = llm.invoke(prompt).strip()

    print(f"[RESPONSE] Generated answer ({len(answer)} chars)\n")

    session["history"].append((question, answer))
    print(f"[CHAT HISTORY] Updated for topic '{new_topic}'\n")

    return answer

def refresh_web_data():
    web_chunks = load_web_chunks()
    if web_chunks:
        web_vector_store.reset_collection()
        web_vector_store.add_documents(web_chunks)
