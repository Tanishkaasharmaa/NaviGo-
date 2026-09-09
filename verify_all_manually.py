import sys
import json
import asyncio
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend import run_travel_agent, benchmark_travel_agent
from verification.existence_gate import resolve_place
from verification.constraints import run_all_checks

def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def main():
    print_section("NaviGo Manual Feature Inspection Tool")

    sample_query = "Plan a 3 days trip to Paris from London under $600 with budget hotels and sightseeing."
    print(f"\n[Test Query]: '{sample_query}'")

    print("\nRunning NaviGo agent graph (LLM + Existence Gate + Verifier)...")
    res = run_travel_agent(sample_query)

    print_section("1. Supervisor Extraction & Constraints")
    print(json.dumps(res.get("trip_constraints", {}), indent=2))
    print(f"\nSelected Agents: {res.get('selected_agents')}")
    print(f"Supervisor Reasoning: {res.get('supervisor_reasoning')}")

    print_section("2. Feature 1 & 2: Existence Gate & Geocoded Stops")
    itinerary_json = res.get("itinerary_json", {})
    days = itinerary_json.get("days", [])
    hotel = itinerary_json.get("hotel", {})

    print(f"Hotel Accommodation: {hotel.get('name')}")
    print(f"  - Verified: {hotel.get('verified')} {'✓' if hotel.get('verified') else '⚠️'}")
    print(f"  - Coordinates: ({hotel.get('lat')}, {hotel.get('lon')})")
    print(f"  - Place ID / Source: {hotel.get('place_id')}")

    print("\nDay-by-Day Geocoded Stops:")
    for d in days:
        print(f"\n  Day {d.get('day')}:")
        for stop in d.get("stops", []):
            status = "Verified ✓" if stop.get("verified") else f"⚠️ Unverified ({stop.get('unverified_reason')})"
            coords = f"({stop.get('lat')}, {stop.get('lon')})" if stop.get("lat") else "(No Coords)"
            print(f"    - [{stop.get('start_time')} - {stop.get('end_time')}] {stop.get('name')} | {coords} | {status} | Cost: ${stop.get('cost')}")

    print_section("3. Feature 1: Deterministic Verifier Report")
    report = res.get("verification_report", {})
    score = report.get("score", 0.0)
    print(f"Constraint Satisfaction Score: {int(score * 100)}%")
    print(f"Checks Performed: {report.get('checks_run')}")
    print(f"Itinerary Retry Attempts: {res.get('itinerary_retry_count', 0)}")

    violations = report.get("violations", [])
    if violations:
        print(f"\nDetected Violations ({len(violations)}):")
        for idx, v in enumerate(violations, 1):
            print(f"  {idx}. [{v.get('category').upper()}] Day {v.get('day')}: {v.get('description')}")
    else:
        print("\nAll hard constraints passed cleanly (0 violations)!")

    print_section("4. Feature 4: Sources & Citation Provenance")
    sources = res.get("sources", [])
    if sources:
        for idx, s in enumerate(sources, 1):
            print(f"  {idx}. {s.get('title')}")
            print(f"     URL: {s.get('url')}")
            print(f"     Agent: {s.get('used_by')} | Timestamp: {s.get('fetched_at')}\n")
    else:
        print("  No external sources recorded.")

    print_section("5. Draft Itinerary Narrative (First 300 chars)")
    print(str(res.get("itinerary", ""))[:300] + "...\n")

if __name__ == "__main__":
    main()
