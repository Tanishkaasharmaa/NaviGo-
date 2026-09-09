import re
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Any
from verification.travel_time import osrm_duration_seconds


class Violation(TypedDict):
    category: str  # "budget", "travel_time", "opening_hours", "hotel_dates", "daily_hours"
    day: int | None
    description: str
    severity: str  # "high", "medium"


def _parse_minutes(time_str: str) -> int | None:
    """Parse '09:00', '9:30 AM', '14:15' into minutes from midnight."""
    if not time_str or not isinstance(time_str, str):
        return None

    cleaned = time_str.strip().upper()

    # Match 24h format (14:30 or 09:00)
    match_24 = re.match(r"^(\d{1,2}):(\d{2})$", cleaned)
    if match_24:
        h, m = int(match_24.group(1)), int(match_24.group(2))
        return h * 60 + m

    # Match 12h format (9:30 AM / 2:15 PM)
    match_12 = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)$", cleaned)
    if match_12:
        h, m, ampm = int(match_12.group(1)), int(match_12.group(2)), match_12.group(3)
        if ampm == "PM" and h < 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
        return h * 60 + m

    return None


def _extract_number(val: Any) -> float | None:
    """Extract numeric value from str/int/float, handling commas/currency symbols."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).replace(",", "")
    matches = re.findall(r"\d+(?:\.\d+)?", s)
    if matches:
        return float(matches[0])
    return None


def check_budget(itinerary_json: dict, stated_budget: Any) -> list[Violation]:
    violations: list[Violation] = []
    budget_limit = _extract_number(stated_budget)
    if budget_limit is None or budget_limit <= 0:
        return violations

    total_cost = _extract_number(itinerary_json.get("total_cost"))
    if total_cost is None:
        # Sum up individual stop costs and hotel cost
        sum_cost = 0.0
        for day in itinerary_json.get("days", []):
            for stop in day.get("stops", []):
                c = _extract_number(stop.get("cost"))
                if c:
                    sum_cost += c
        hotel = itinerary_json.get("hotel", {})
        h_cost = _extract_number(hotel.get("total_cost") or hotel.get("cost"))
        if h_cost:
            sum_cost += h_cost
        total_cost = sum_cost

    if total_cost > budget_limit:
        violations.append({
            "category": "budget",
            "day": None,
            "description": (
                f"Total itinerary cost ({total_cost:.2f}) exceeds stated budget "
                f"limit of {budget_limit:.2f}."
            ),
            "severity": "high",
        })

    return violations


async def check_travel_time(days_list: list[dict]) -> list[Violation]:
    violations: list[Violation] = []

    for day_info in days_list:
        day_num = day_info.get("day", 1)
        stops = day_info.get("stops", [])

        for i in range(len(stops) - 1):
            prev_stop = stops[i]
            next_stop = stops[i + 1]

            prev_end = _parse_minutes(prev_stop.get("end_time"))
            next_start = _parse_minutes(next_stop.get("start_time"))

            if prev_end is None or next_start is None:
                continue

            gap_minutes = next_start - prev_end

            # Only check OSRM travel time if both stops have valid coordinates
            p_lat, p_lon = prev_stop.get("lat"), prev_stop.get("lon")
            n_lat, n_lon = next_stop.get("lat"), next_stop.get("lon")

            if p_lat is not None and p_lon is not None and n_lat is not None and n_lon is not None:
                duration_sec = await osrm_duration_seconds(p_lat, p_lon, n_lat, n_lon)
                duration_min = math_ceil_min(duration_sec)

                # Buffer check: if scheduled gap is less than travel duration
                if gap_minutes < duration_min:
                    violations.append({
                        "category": "travel_time",
                        "day": day_num,
                        "description": (
                            f"Day {day_num}: Travel time from '{prev_stop.get('name')}' to "
                            f"'{next_stop.get('name')}' requires ~{duration_min} mins travel time, "
                            f"but the schedule gap is only {gap_minutes} mins."
                        ),
                        "severity": "high",
                    })

    return violations


def math_ceil_min(sec: float) -> int:
    import math
    return math.ceil(sec / 60.0)


def check_hotel_dates(itinerary_json: dict, total_days: int) -> list[Violation]:
    violations: list[Violation] = []
    hotel = itinerary_json.get("hotel")
    if not hotel or not isinstance(hotel, dict):
        violations.append({
            "category": "hotel_dates",
            "day": 1,
            "description": "No hotel accommodation entry found in itinerary.",
            "severity": "high",
        })
        return violations

    check_in = hotel.get("check_in_day")
    check_out = hotel.get("check_out_day")

    if check_in is not None and check_in != 1:
        violations.append({
            "category": "hotel_dates",
            "day": 1,
            "description": f"Hotel check-in day is set to Day {check_in}, expected Day 1.",
            "severity": "medium",
        })

    if check_out is not None and total_days > 1 and check_out < total_days:
        violations.append({
            "category": "hotel_dates",
            "day": total_days,
            "description": (
                f"Hotel check-out day is Day {check_out}, but the trip duration is "
                f"{total_days} days."
            ),
            "severity": "medium",
        })

    return violations


def check_daily_hours(days_list: list[dict], ceiling_hours: float = 10.0) -> list[Violation]:
    violations: list[Violation] = []

    for day_info in days_list:
        day_num = day_info.get("day", 1)
        stops = day_info.get("stops", [])

        if not stops:
            continue

        start_times = [_parse_minutes(s.get("start_time")) for s in stops if _parse_minutes(s.get("start_time")) is not None]
        end_times = [_parse_minutes(s.get("end_time")) for s in stops if _parse_minutes(s.get("end_time")) is not None]

        if not start_times or not end_times:
            continue

        day_span_hours = (max(end_times) - min(start_times)) / 60.0

        if day_span_hours > ceiling_hours:
            violations.append({
                "category": "daily_hours",
                "day": day_num,
                "description": (
                    f"Day {day_num} total activity span is {day_span_hours:.1f} hours, "
                    f"exceeding the sane ceiling of {ceiling_hours:.1f} hours."
                ),
                "severity": "medium",
            })

    return violations


async def run_all_checks(
    itinerary_json: dict,
    trip_constraints: dict,
    ceiling_hours: float = 10.0
) -> dict:
    """
    Runs all deterministic constraint checks against itinerary_json.
    Returns {
        "violations": list[Violation],
        "score": float (0.0 to 1.0),
        "checks_run": list[str]
    }
    """
    violations: list[Violation] = []
    days = itinerary_json.get("days", [])
    total_days = len(days)

    # 1. Budget Check
    budget_val = trip_constraints.get("budget")
    v_budget = check_budget(itinerary_json, budget_val)
    violations.extend(v_budget)

    # 2. Travel Time Check (Async OSRM)
    v_travel = await check_travel_time(days)
    violations.extend(v_travel)

    # 3. Hotel Dates Check
    v_hotel = check_hotel_dates(itinerary_json, total_days)
    violations.extend(v_hotel)

    # 4. Daily Activity Ceiling Check
    v_daily = check_daily_hours(days, ceiling_hours=ceiling_hours)
    violations.extend(v_daily)

    # Calculate Constraint Satisfaction Score
    # Base 1.0 minus penalty per violation
    penalty = len(violations) * 0.2
    score = max(0.0, round(1.0 - penalty, 2))

    return {
        "violations": violations,
        "score": score,
        "checks_run": ["budget", "travel_time", "hotel_dates", "daily_hours"],
    }
