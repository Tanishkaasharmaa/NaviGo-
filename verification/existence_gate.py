import os
import time
import asyncio
import httpx
from datetime import datetime
from rapidfuzz import fuzz

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()

# In-memory cache for place resolutions during run
_RESOLVE_CACHE: dict[str, dict] = {}
_NOMINATIM_LOCK = asyncio.Lock()
_LAST_NOMINATIM_CALL = 0.0


async def resolve_place(name: str, city_context: str = "") -> dict:
    """
    Resolves a named attraction/hotel against Google Places API or Nominatim OSM.
    Returns:
        {
            "name": str,
            "resolved_name": str,
            "lat": float | None,
            "lon": float | None,
            "verified": bool,
            "place_id": str | None,
            "opening_hours": dict | None,
            "source": dict | None,
            "unverified_reason": str | None
        }
    """
    clean_name = name.strip()
    if not clean_name:
        return {
            "name": name,
            "resolved_name": name,
            "lat": None,
            "lon": None,
            "verified": False,
            "place_id": None,
            "opening_hours": None,
            "source": None,
            "unverified_reason": "Empty place name",
        }

    cache_key = f"{clean_name.lower()}::{city_context.lower()}"
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]

    result = None

    # 1. Try Google Places Text Search if API key available
    if GOOGLE_PLACES_API_KEY:
        try:
            result = await _resolve_google_places(clean_name, city_context)
        except Exception as exc:
            print(f"[ExistenceGate] Google Places lookup failed for '{clean_name}': {exc}")

    # 2. Fallback to Nominatim OpenStreetMap
    if not result or not result.get("verified"):
        try:
            result = await _resolve_nominatim(clean_name, city_context)
        except Exception as exc:
            print(f"[ExistenceGate] Nominatim lookup failed for '{clean_name}': {exc}")
            result = {
                "name": clean_name,
                "resolved_name": clean_name,
                "lat": None,
                "lon": None,
                "verified": False,
                "place_id": None,
                "opening_hours": None,
                "source": None,
                "unverified_reason": f"Geocoding service error: {exc}",
            }

    _RESOLVE_CACHE[cache_key] = result
    return result


async def _resolve_google_places(name: str, city_context: str) -> dict:
    query = f"{name} {city_context}".strip()
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY}

    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return {
            "name": name,
            "resolved_name": name,
            "lat": None,
            "lon": None,
            "verified": False,
            "place_id": None,
            "opening_hours": None,
            "source": None,
            "unverified_reason": "No place found in Google Places",
        }

    first = results[0]
    display_name = first.get("name", "")
    loc = first.get("geometry", {}).get("location", {})
    lat = loc.get("lat")
    lon = loc.get("lng")
    place_id = first.get("place_id")

    # Score match confidence
    sim = fuzz.partial_ratio(name.lower(), display_name.lower())
    if sim < 50:
        return {
            "name": name,
            "resolved_name": display_name,
            "lat": lat,
            "lon": lon,
            "verified": False,
            "place_id": place_id,
            "opening_hours": first.get("opening_hours"),
            "source": None,
            "unverified_reason": f"Low match confidence ({sim}%) with '{display_name}'",
        }

    place_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else ""
    return {
        "name": name,
        "resolved_name": display_name,
        "lat": lat,
        "lon": lon,
        "verified": True,
        "place_id": place_id,
        "opening_hours": first.get("opening_hours"),
        "source": {
            "title": f"Google Maps: {display_name}",
            "url": place_url,
            "fetched_at": datetime.utcnow().isoformat(),
            "used_by": "existence_gate",
        },
        "unverified_reason": None,
    }


async def _resolve_nominatim(name: str, city_context: str) -> dict:
    global _LAST_NOMINATIM_CALL
    query = f"{name}, {city_context}".strip(", ")
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "NaviGo-Travel-Planner/1.0 (contact@navigo.app)"
    }
    params = {
        "q": query,
        "format": "json",
        "addressdetails": "1",
        "limit": "1",
    }

    # Respect Nominatim 1 req/sec policy
    async with _NOMINATIM_LOCK:
        now = time.time()
        elapsed = now - _LAST_NOMINATIM_CALL
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        _LAST_NOMINATIM_CALL = time.time()

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return {
                    "name": name,
                    "resolved_name": name,
                    "lat": None,
                    "lon": None,
                    "verified": False,
                    "place_id": None,
                    "opening_hours": None,
                    "source": None,
                    "unverified_reason": f"OSM returned status HTTP {resp.status_code}",
                }
            items = resp.json()

    if not items:
        return {
            "name": name,
            "resolved_name": name,
            "lat": None,
            "lon": None,
            "verified": False,
            "place_id": None,
            "opening_hours": None,
            "source": None,
            "unverified_reason": f"Place '{name}' could not be resolved via OpenStreetMap",
        }

    first = items[0]
    display_name = first.get("display_name", "")
    try:
        lat = float(first.get("lat"))
        lon = float(first.get("lon"))
    except (TypeError, ValueError):
        lat, lon = None, None

    osm_id = first.get("osm_id")
    osm_type = first.get("osm_type")

    # Score match confidence against attraction name
    sim = fuzz.token_set_ratio(name.lower(), display_name.lower())
    if sim < 40 and lat is None:
        return {
            "name": name,
            "resolved_name": display_name,
            "lat": lat,
            "lon": lon,
            "verified": False,
            "place_id": str(osm_id) if osm_id else None,
            "opening_hours": None,
            "source": None,
            "unverified_reason": f"Low match confidence ({sim}%) with '{display_name[:40]}'",
        }

    osm_url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_id and osm_type else "https://www.openstreetmap.org"

    return {
        "name": name,
        "resolved_name": display_name,
        "lat": lat,
        "lon": lon,
        "verified": True,
        "place_id": str(osm_id) if osm_id else None,
        "opening_hours": None,  # Nominatim standard search doesn't return full schedule
        "source": {
            "title": f"OpenStreetMap: {name}",
            "url": osm_url,
            "fetched_at": datetime.utcnow().isoformat(),
            "used_by": "existence_gate",
        },
        "unverified_reason": None,
    }
