import os
import json
from datetime import datetime
from typing import Any
from benchmark.adapter import evaluate_itinerary_result

def print_realtime_node_update(node_name: str, update: dict[str, Any]) -> None:
    """Prints a real-time event log as each graph step executes."""
    now_str = datetime.now().strftime("%H:%M:%S")

    if node_name == "guardrail_node":
        allowed = update.get("guardrail_allowed", True)
        reason = update.get("guardrail_reason", "Passed topic check")
        status_icon = "✅" if allowed else "🛑"
        print(f"\n[{now_str}] 🛡️ [GUARDRAIL NODE] {status_icon} Allowed: {allowed}")
        if reason:
            print(f"            • Reason: {reason}")

    elif node_name == "supervisor_agent":
        agents = update.get("selected_agents", [])
        constraints = update.get("trip_constraints", {})
        reasoning = update.get("supervisor_reasoning", "")
        print(f"\n[{now_str}] 🧭 [SUPERVISOR AGENT] Request analyzed & planned")
        print(f"            • Selected Specialists: {', '.join(agents) if agents else 'None'}")
        if constraints:
            print(f"            • Trip Constraints   : {json.dumps(constraints)}")
        if reasoning:
            print(f"            • Reasoning          : {reasoning}")

    elif node_name == "flight_agent":
        results = update.get("flight_results", "")
        summary = results.strip().split("\n")[0] if results else "Flight search completed."
        print(f"\n[{now_str}] ✈️ [FLIGHT AGENT] Search completed")
        print(f"            • Result Summary: {summary[:120]}")

    elif node_name == "hotel_agent":
        results = update.get("hotel_results", "")
        summary = results.strip().split("\n")[0] if results else "Hotel search completed."
        print(f"\n[{now_str}] 🏨 [HOTEL AGENT] Search completed")
        print(f"            • Result Summary: {summary[:120]}")

    elif node_name == "weather_agent":
        results = update.get("weather_results", "")
        summary = results.strip().split("\n")[0] if results else "Weather lookup completed."
        print(f"\n[{now_str}] 🌤️ [WEATHER AGENT] Lookup completed")
        print(f"            • Result Summary: {summary[:120]}")

    elif node_name == "budget_agent":
        results = update.get("budget_results", "")
        summary = results.strip().split("\n")[0] if results else "Budget allocation completed."
        print(f"\n[{now_str}] 💰 [BUDGET AGENT] Budget analysis completed")
        print(f"            • Result Summary: {summary[:120]}")

    elif node_name == "itinerary_agent":
        itinerary_json = update.get("itinerary_json", {})
        dest = itinerary_json.get("destination", "N/A")
        cost = itinerary_json.get("total_cost", "N/A")
        days = len(itinerary_json.get("days", []))
        print(f"\n[{now_str}] 🗺️ [ITINERARY AGENT] Draft itinerary generated")
        print(f"            • Destination : {dest}")
        print(f"            • Cost Est.   : ${cost}")
        print(f"            • Days Count  : {days} days")

    elif node_name == "itinerary_verifier":
        report = update.get("verification_report", {})
        retry_count = update.get("itinerary_retry_count", 0)
        score = report.get("score", 1.0)
        violations = report.get("violations", [])
        status_icon = "✅" if not violations else "⚠️"
        print(f"\n[{now_str}] 🔬 [ITINERARY VERIFIER] {status_icon} Verification complete (Attempt #{retry_count + 1})")
        print(f"            • Quality Score: {score * 100:.1f}%")
        print(f"            • Violations   : {len(violations)} detected")
        for v in violations:
            print(f"              - [{v.get('category','').upper()}] {v.get('description','')}")


def print_analysis_report(result: dict[str, Any], query: str = "") -> None:
    """
    Prints a detailed analytical breakdown of the test execution, LLM outputs,
    place existence verification, constraint satisfaction, and benchmark evaluation.
    """
    model_name = os.getenv("GROQ_MODEL", "Unknown Model")
    query_text = query or result.get("user_query", "N/A")
    status = result.get("status", "complete")
    answer = result.get("answer", "")
    itinerary_json = result.get("itinerary_json") or {}
    verification_report = result.get("verification_report") or {}
    violations = verification_report.get("violations", [])
    score = verification_report.get("score", 1.0)
    selected_agents = result.get("selected_agents", [])
    trip_constraints = result.get("trip_constraints", {})
    llm_calls = result.get("llm_calls", 0)
    retry_count = result.get("itinerary_retry_count", 0)
    sources = result.get("sources", [])

    # Evaluate against TravelPlanner benchmark criteria
    eval_metrics = evaluate_itinerary_result({"query": query_text}, result)

    print("\n" + "=" * 80)
    print(" 📊 NAVIGO TEST EXECUTION & FEATURE ANALYSIS REPORT ")
    print("=" * 80)

    # 1. Test Details & Setup
    print("\n🔹 1. TEST CONFIGURATION & EXECUTION DETAILS")
    print(f"   • User Query       : '{query_text}'")
    print(f"   • LLM Model Used   : {model_name}")
    print(f"   • Selected Agents  : {', '.join(selected_agents) if selected_agents else 'None'}")
    print(f"   • Trip Constraints : {json.dumps(trip_constraints)}")
    print(f"   • Total LLM Calls  : {llm_calls}")
    print(f"   • Refinement Retries: {retry_count}")
    print(f"   • Execution Status : {status}")

    # 2. LLM Output Analysis
    print("\n🤖 2. LLM GENERATED OUTPUT")
    print("   --- [ Narrative Answer Excerpt ] ---")
    lines = answer.strip().split("\n")
    preview = "\n".join(lines[:10])
    if len(lines) > 10:
        preview += f"\n   ... ({len(lines) - 10} more lines omitted)"
    print(f"   {preview}\n")

    print("   --- [ Structured JSON Breakdown ] ---")
    if itinerary_json:
        dest = itinerary_json.get("destination", "N/A")
        total_cost = itinerary_json.get("total_cost", "N/A")
        hotel = itinerary_json.get("hotel", {})
        days = itinerary_json.get("days", [])

        print(f"   • Destination : {dest}")
        print(f"   • Total Cost  : ${total_cost}")
        print(f"   • Hotel       : {hotel.get('name', 'N/A')} (Check-in: Day {hotel.get('check_in_day', 'N/A')}, Check-out: Day {hotel.get('check_out_day', 'N/A')})")
        print(f"   • Days Plan   : {len(days)} days generated")

        verified_places = []
        unverified_places = []

        for day_info in days:
            stops = day_info.get("stops", [])
            for stop in stops:
                name = stop.get("name", "Unknown Stop")
                if stop.get("verified"):
                    verified_places.append(f"{name} ({stop.get('lat', 0):.4f}, {stop.get('lon', 0):.4f})")
                else:
                    unverified_places.append(f"{name} (Reason: {stop.get('unverified_reason', 'Not verified in OSM')})")

        print(f"\n   📍 Existence Gate Resolution (OpenStreetMap/Overpass):")
        print(f"      • Verified Places ({len(verified_places)})   : {', '.join(verified_places) if verified_places else 'None'}")
        if unverified_places:
            print(f"      • Unverified/Hallucinated ({len(unverified_places)}): {'; '.join(unverified_places)}")
    else:
        print("   ⚠️ No structured itinerary JSON was generated by LLM.")

    # 3. Constraint Satisfaction Analysis
    print("\n🔍 3. CONSTRAINT SATISFACTION & VERIFICATION REPORT")
    print(f"   • Overall Quality Score : {score * 100:.1f}% ({score:.2f} / 1.0)")
    print(f"   • Total Violations Found: {len(violations)}")

    if violations:
        print("\n   ⚠️ DETECTED VIOLATIONS:")
        for idx, v in enumerate(violations, 1):
            category = v.get("category", "general").upper()
            severity = v.get("severity", "medium").upper()
            desc = v.get("description", "")
            day_str = f" [Day {v['day']}]" if v.get("day") else ""
            print(f"     {idx}. [{category} - {severity}]{day_str} {desc}")
    else:
        print("   ✅ ALL CONSTRAINTS SATISFIED cleanly! No budget, travel-time, or date violations.")

    # 4. Benchmark Metric Summary
    print("\n📈 4. BENCHMARK & EVALUATION METRICS")
    print(f"   • Delivery Status    : {'✅ DELIVERED' if eval_metrics['delivered'] else '❌ FAILED'}")
    print(f"   • Commonsense Pass   : {'✅ PASSED' if eval_metrics['commonsense_pass'] else '❌ FAILED'}")
    print(f"   • Hard Constraint Pass: {'✅ PASSED' if eval_metrics['hard_constraint_pass'] else '❌ FAILED'}")
    print(f"   • Final Evaluation   : {'✅ PASS' if eval_metrics['final_pass'] else '❌ FAIL'}")

    if sources:
        print(f"\n   🔗 Information Sources Used: {len(sources)} sources (e.g. {sources[0].get('url', 'N/A') if sources else ''})")

    print("=" * 80 + "\n")
