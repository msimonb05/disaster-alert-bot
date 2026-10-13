from typing import Optional, Dict
from functools import lru_cache
from model.services.pagasa_bulletin_parser import get_latest_bulletin, extract_tcws_impact
import logging
import time

logger = logging.getLogger(__name__)

def fetch_weather_warning() -> Optional[Dict]:
    return _cached_weather_warning(int(time.time() // 600))

# Reuses the same bulletin for 10 minutes. A failed fetch also stays cached for that long.
@lru_cache(maxsize=1)
def _cached_weather_warning(time_window: int) -> Optional[Dict]:
    try:
        bulletin = get_latest_bulletin()
        if not bulletin:
            return None

        tcws_list = bulletin.get("tcws") or []
        highest_signal = max(tcws_list, key=lambda s: s.get("signal_no", 0)) if tcws_list else None
        location = bulletin.get("location_of_center", {})

        return {
            "classification": bulletin.get("classification"),
            "name": bulletin.get("name"),
            "issued": bulletin.get("issued"),
            "intensity": bulletin.get("intensity"),
            "present_movement": bulletin.get("present_movement"),
            "location": {
                "description": location.get("description"),
                "as_of": location.get("as_of")
            },
            "highest_tcws": {
                "signal_no": highest_signal.get("signal_no") if highest_signal else None,
                "affected_areas": highest_signal.get("affected_areas") if highest_signal else None,
                "impact": extract_tcws_impact(highest_signal["signal_no"]) if highest_signal else "Unknown"
            },
            "tcws": [
                {**signal, "impact": extract_tcws_impact(signal.get("signal_no"))}
                for signal in tcws_list
            ],
            "source": bulletin.get("source")
        }

    except Exception as e:
        logger.exception("Failed to fetch weather warning")
        return None
