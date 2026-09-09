from typing import Any


def evaluate_itinerary_result(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluates NaviGo's output against TravelPlanner benchmark criteria.
    Returns:
        {
            "delivered": bool,
            "commonsense_pass": bool,
            "hard_constraint_pass": bool,
            "final_pass": bool,
            "violations_count": int,
            "score": float
        }
    """
    answer = result.get("answer", "")
    itinerary_json = result.get("itinerary_json", {})
    verification_report = result.get("verification_report", {})
    violations = verification_report.get("violations", [])

    days = itinerary_json.get("days", [])

    # 1. Delivery Rate check: Was a parseable structured itinerary generated?
    delivered = bool(days and len(days) > 0 and len(answer) > 50)

    if not delivered:
        return {
            "delivered": False,
            "commonsense_pass": False,
            "hard_constraint_pass": False,
            "final_pass": False,
            "violations_count": len(violations) + 1,
            "score": 0.0,
        }

    # 2. Commonsense Pass check: No daily activity ceiling or travel time buffer violations
    commonsense_violations = [
        v for v in violations if v.get("category") in ("daily_hours", "travel_time")
    ]
    commonsense_pass = (len(commonsense_violations) == 0)

    # 3. Hard Constraint Pass check: Budget and hotel check-in/out constraints
    hard_violations = [
        v for v in violations if v.get("category") in ("budget", "hotel_dates")
    ]
    hard_constraint_pass = (len(hard_violations) == 0)

    # 4. Final Pass: Passes all checks
    final_pass = (delivered and commonsense_pass and hard_constraint_pass)

    score = float(verification_report.get("score", 1.0 if final_pass else 0.5))

    return {
        "delivered": delivered,
        "commonsense_pass": commonsense_pass,
        "hard_constraint_pass": hard_constraint_pass,
        "final_pass": final_pass,
        "violations_count": len(violations),
        "score": score,
    }
