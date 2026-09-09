import math
import httpx


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


async def osrm_duration_seconds(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    mode: str = "driving"
) -> float:
    """
    Query OSRM API for routing travel time in seconds between two points.
    Applies walking mode heuristic when distance < 1.5 km.
    Falls back gracefully if OSRM endpoint is unreachable.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0

    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    if dist_km < 0.05:  # Virtually same location (< 50 meters)
        return 0.0

    # Apply heuristic: walking for short distances (< 1.5km)
    actual_mode = "walking" if dist_km < 1.5 else mode

    url = (
        f"https://router.project-osrm.org/route/v1/{actual_mode}/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=false"
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes", [])
                if routes and "duration" in routes[0]:
                    return float(routes[0]["duration"])
    except Exception as exc:
        print(f"[TravelTime] OSRM query failed ({actual_mode}): {exc}. Using distance heuristic fallback.")

    # Fallback duration calculation (Urban speed estimates)
    # Walking: ~4.5 km/h = 1.25 m/s
    # Driving (urban traffic): ~25 km/h = 6.94 m/s
    speed_kmh = 4.5 if actual_mode == "walking" else 25.0
    duration_hours = dist_km / speed_kmh
    return duration_hours * 3600.0
