import math
import re
import json
from typing import Any, Dict, List, Tuple, Optional
import requests

OSRM_API_URL = "http://router.project-osrm.org/route/v1/driving/{lng1},{lat1};{lng2},{lat2}?overview=false"
REQUEST_TIMEOUT_SECONDS = 3


def time_to_minutes(time_str: str) -> Optional[int]:
    """Convert time strings like '09:30', '9:30', '14:00', '2:00 PM' to minutes from midnight."""
    if not time_str or not isinstance(time_str, str):
        return None

    cleaned = time_str.strip().upper()
    match = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", cleaned)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    period = match.group(3)

    if period == "PM" and hours < 12:
        hours += 12
    elif period == "AM" and hours == 12:
        hours = 0

    return hours * 60 + minutes


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance in kilometers between two GPS points."""
    R = 6371.0  # Earth radius in km
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


def estimate_travel_time_osrm_or_haversine(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> int:
    """
    Estimate travel time in minutes between two coordinates.
    Attempts OSRM API call first; falls back to Haversine distance calculation.
    """
    # If points are identical or practically zero distance
    if abs(lat1 - lat2) < 0.0001 and abs(lon1 - lon2) < 0.0001:
        return 0

    try:
        url = OSRM_API_URL.format(lng1=lon2, lat1=lat1, lng2=lon1, lat2=lat2)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            routes = data.get("routes", [])
            if routes:
                duration_seconds = routes[0].get("duration", 0)
                duration_minutes = int(math.ceil(duration_seconds / 60.0))
                return duration_minutes
    except Exception as exc:
        print(f"[Verifier] OSRM fallback triggered: {exc}")

    # Fallback: Haversine distance at ~30 km/h average urban speed + 5 min buffer
    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    estimated_mins = int(math.ceil((dist_km / 30.0) * 60 + 5))
    return max(5, estimated_mins)


def parse_numeric_budget(budget_str: str) -> Optional[float]:
    """Parse budget limits like '$1200', '1200 USD', '2 lakhs', '200000' into a float number."""
    if not budget_str or not isinstance(budget_str, str):
        return None

    cleaned = budget_str.lower().strip()
    
    # Handle "lakh" or "lakhs" (1 lakh = 100,000)
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh", cleaned)
    if lakh_match:
        return float(lakh_match.group(1)) * 100000.0

    # Handle standard numbers
    numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", cleaned)
    if numbers:
        val = numbers[0].replace(",", "")
        try:
            return float(val)
        except ValueError:
            return None
    return None


def parse_structured_itinerary(llm_output: str) -> List[Dict[str, Any]]:
    """Extract and parse structured itinerary JSON from LLM output."""
    if not llm_output or not isinstance(llm_output, str):
        return []

    # Look for explicitly tagged JSON blocks first
    json_match = re.search(r"<json_itinerary>\s*(\[.*?\]|\{.*?\})\s*</json_itinerary>", llm_output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fall back to markdown ```json ... ``` code blocks
        code_block_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\"stops\".*?\})\s*```", llm_output, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Fall back to finding raw JSON array
            raw_array_match = re.search(r"\[\s*\{.*\}\s*\]", llm_output, re.DOTALL)
            json_str = raw_array_match.group(0) if raw_array_match else ""

    if not json_str:
        return []

    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "stops" in data:
            stops = data["stops"]
        elif isinstance(data, list):
            stops = data
        else:
            stops = []

        validated_stops = []
        for stop in stops:
            if not isinstance(stop, dict):
                continue
            validated_stops.append({
                "day": int(stop.get("day", 1)),
                "stop_name": str(stop.get("stop_name", "Activity")),
                "coordinates": stop.get("coordinates", {"lat": 0.0, "lng": 0.0}),
                "start_time": str(stop.get("start_time", "09:00")),
                "end_time": str(stop.get("end_time", "10:00")),
                "cost": float(stop.get("cost", 0.0)),
                "opening_hours": stop.get("opening_hours"),
                "stop_type": str(stop.get("stop_type", "activity")).lower()
            })
        return validated_stops
    except Exception as exc:
        print(f"[Verifier] JSON parsing error: {exc}")
        return []


def check_budget_constraint(
    stops: List[Dict[str, Any]],
    user_budget_raw: Any,
    user_query: str
) -> Tuple[bool, Optional[str], float, float]:
    """
    Constraint 1: Sum of costs against stated budget limit.
    Returns: (is_passed, violation_message_or_none, total_cost, budget_limit)
    """
    total_stop_cost = sum(stop.get("cost", 0.0) for stop in stops)

    # Resolve budget limit from constraints or query
    budget_limit = None
    if user_budget_raw:
        budget_limit = parse_numeric_budget(str(user_budget_raw))
    if budget_limit is None and user_query:
        budget_limit = parse_numeric_budget(user_query)

    if budget_limit is None or budget_limit <= 0:
        # Budget constraint not explicitly specified or unparseable
        return True, None, total_stop_cost, 0.0

    # Allow a 5% margin before failing budget constraint
    if total_stop_cost > (budget_limit * 1.05):
        over_amount = total_stop_cost - budget_limit
        msg = (
            f"Budget Exceeded: Total scheduled itinerary cost (${total_stop_cost:,.2f}) "
            f"exceeds stated budget (${budget_limit:,.2f}) by ${over_amount:,.2f}."
        )
        return False, msg, total_stop_cost, budget_limit

    return True, None, total_stop_cost, budget_limit


def check_travel_time_constraint(stops: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Constraint 2: Travel time between consecutive stops on the same day.
    Flags when gap between stop A end_time and stop B start_time is shorter than required travel time.
    """
    violations = []
    
    # Group stops by day
    days: Dict[int, List[Dict[str, Any]]] = {}
    for stop in stops:
        day = stop["day"]
        days.setdefault(day, []).append(stop)

    for day, day_stops in days.items():
        # Sort stops by start_time
        sorted_stops = sorted(
            day_stops,
            key=lambda s: time_to_minutes(s["start_time"]) or 0
        )

        for i in range(len(sorted_stops) - 1):
            curr_stop = sorted_stops[i]
            next_stop = sorted_stops[i + 1]

            curr_end_min = time_to_minutes(curr_stop["end_time"])
            next_start_min = time_to_minutes(next_stop["start_time"])

            if curr_end_min is None or next_start_min is None:
                continue

            available_gap = next_start_min - curr_end_min
            
            # Extract coordinates
            c1 = curr_stop.get("coordinates", {})
            c2 = next_stop.get("coordinates", {})
            
            lat1 = c1.get("lat") if isinstance(c1, dict) else (c1[0] if isinstance(c1, list) and len(c1) > 0 else 0.0)
            lon1 = c1.get("lng") if isinstance(c1, dict) else (c1[1] if isinstance(c1, list) and len(c1) > 1 else 0.0)
            lat2 = c2.get("lat") if isinstance(c2, dict) else (c2[0] if isinstance(c2, list) and len(c2) > 0 else 0.0)
            lon2 = c2.get("lng") if isinstance(c2, dict) else (c2[1] if isinstance(c2, list) and len(c2) > 1 else 0.0)

            # Skip coordinate travel check if coordinates are invalid/default
            if (lat1 == 0.0 and lon1 == 0.0) or (lat2 == 0.0 and lon2 == 0.0):
                required_travel_min = 15  # Default nominal buffer
            else:
                required_travel_min = estimate_travel_time_osrm_or_haversine(lat1, lon1, lat2, lon2)

            if available_gap < required_travel_min:
                shortfall = required_travel_min - available_gap
                msg = (
                    f"Travel Time Conflict (Day {day}): Gap between '{curr_stop['stop_name']}' "
                    f"(ends {curr_stop['end_time']}) and '{next_stop['stop_name']}' "
                    f"(starts {next_stop['start_time']}) is {max(0, available_gap)} mins, "
                    f"but transit requires at least {required_travel_min} mins (shortfall: {shortfall} mins)."
                )
                violations.append(msg)

    return len(violations) == 0, violations


def check_opening_hours_constraint(stops: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Constraint 3: Opening hours against scheduled slot.
    """
    violations = []
    for stop in stops:
        op_hours = stop.get("opening_hours")
        if not op_hours or not isinstance(op_hours, dict):
            continue

        open_str = op_hours.get("open")
        close_str = op_hours.get("close")
        if not open_str or not close_str:
            continue

        open_min = time_to_minutes(open_str)
        close_min = time_to_minutes(close_str)
        start_min = time_to_minutes(stop["start_time"])
        end_min = time_to_minutes(stop["end_time"])

        if None in (open_min, close_min, start_min, end_min):
            continue

        # Check if scheduled slot falls outside operating hours
        if start_min < open_min or end_min > close_min:
            msg = (
                f"Opening Hours Violation (Day {stop['day']}): '{stop['stop_name']}' is scheduled "
                f"from {stop['start_time']} to {stop['end_time']}, but operating hours are {open_str} to {close_str}."
            )
            violations.append(msg)

    return len(violations) == 0, violations


def check_hotel_checkin_constraint(stops: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Constraint 4: Hotel check-in on arrival day (Day 1) and no overlapping hotel check-in bookings across days.
    """
    violations = []
    hotel_stops = [s for s in stops if s.get("stop_type") in ("hotel_checkin", "hotel", "accommodation")]

    if hotel_stops:
        # Check if any hotel check-in exists on Day 1
        day1_hotels = [s for s in hotel_stops if s.get("day") == 1]
        if not day1_hotels:
            violations.append(
                "Hotel Check-in Violation: No hotel check-in or accommodation is scheduled on Day 1 (Arrival Day)."
            )

        # Check for multiple check-ins on same day
        days_with_hotels = [s["day"] for s in hotel_stops]
        if len(days_with_hotels) != len(set(days_with_hotels)):
            violations.append(
                "Hotel Booking Overlap: Multiple hotel check-in slots scheduled on the same day."
            )

    return len(violations) == 0, violations


def check_daily_activity_ceiling(
    stops: List[Dict[str, Any]], max_daily_hours: float = 10.0
) -> Tuple[bool, List[str]]:
    """
    Constraint 5: Total daily activity hours against a sane ceiling (e.g. 10 hours).
    """
    violations = []
    days: Dict[int, float] = {}

    for stop in stops:
        day = stop["day"]
        start_min = time_to_minutes(stop["start_time"])
        end_min = time_to_minutes(stop["end_time"])

        if start_min is not None and end_min is not None and end_min > start_min:
            duration_hours = (end_min - start_min) / 60.0
            days[day] = days.get(day, 0.0) + duration_hours

    for day, total_hours in days.items():
        if total_hours > max_daily_hours:
            over = total_hours - max_daily_hours
            msg = (
                f"Daily Activity Ceiling Exceeded (Day {day}): Total activity is {total_hours:.1f} hrs, "
                f"exceeding maximum ceiling of {max_daily_hours:.1f} hrs per day by {over:.1f} hrs."
            )
            violations.append(msg)

    return len(violations) == 0, violations


def run_deterministic_verifier(
    llm_itinerary_output: str,
    trip_constraints: Dict[str, Any],
    user_query: str
) -> Dict[str, Any]:
    """
    Main verification pipeline executing pure Python deterministic checks across all 5 hard constraints.
    Returns a comprehensive verification report with score, pass/fail status, and structured violations.
    """
    stops = parse_structured_itinerary(llm_itinerary_output)

    # 1. Budget Constraint
    budget_passed, budget_msg, total_cost, budget_limit = check_budget_constraint(
        stops, trip_constraints.get("budget"), user_query
    )
    
    # 2. Travel Time Constraint (OSRM/Haversine)
    travel_passed, travel_violations = check_travel_time_constraint(stops)

    # 3. Opening Hours Constraint
    hours_passed, hours_violations = check_opening_hours_constraint(stops)

    # 4. Hotel Check-in Constraint
    hotel_passed, hotel_violations = check_hotel_checkin_constraint(stops)

    # 5. Daily Activity Ceiling Constraint
    ceiling_passed, ceiling_violations = check_daily_activity_ceiling(stops)

    all_violations: List[str] = []
    if budget_msg:
        all_violations.append(budget_msg)
    all_violations.extend(travel_violations)
    all_violations.extend(hours_violations)
    all_violations.extend(hotel_violations)
    all_violations.extend(ceiling_violations)

    total_checks = 5
    passed_checks = sum([budget_passed, travel_passed, hours_passed, hotel_passed, ceiling_passed])
    
    score = round((passed_checks / float(total_checks)) * 100.0, 1)
    if not stops:
        # If structured JSON could not be parsed, mark as warning score
        score = 60.0
        all_violations.insert(0, "Format Warning: Itinerary JSON structure could not be parsed fully for deterministic check.")

    passed_all = len(all_violations) == 0

    return {
        "passed": passed_all,
        "score": score,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "violations": all_violations,
        "parsed_stops_count": len(stops),
        "checks_breakdown": {
            "budget": {"passed": budget_passed, "total_cost": total_cost, "budget_limit": budget_limit},
            "travel_time": {"passed": travel_passed, "violations_count": len(travel_violations)},
            "opening_hours": {"passed": hours_passed, "violations_count": len(hours_violations)},
            "hotel_checkin": {"passed": hotel_passed, "violations_count": len(hotel_violations)},
            "daily_activity_ceiling": {"passed": ceiling_passed, "violations_count": len(ceiling_violations)}
        }
    }
