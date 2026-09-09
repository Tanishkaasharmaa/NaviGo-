from verification.verifier import run_deterministic_verifier, parse_structured_itinerary

mock_llm_output = """
Here is your detailed 2-day itinerary:

Day 1: Tokyo Arrival & Sightseeing
- Hotel check-in at Hotel Gracery Shinjuku (14:00 - 15:00)
- Visit Senso-ji Temple (15:30 - 17:30)
- Dinner at Shibuya (18:00 - 20:00)

<json_itinerary>
[
  {
    "day": 1,
    "stop_name": "Hotel Gracery Shinjuku Check-in",
    "coordinates": {"lat": 35.6953, "lng": 139.7022},
    "start_time": "14:00",
    "end_time": "15:00",
    "cost": 150.0,
    "stop_type": "hotel_checkin"
  },
  {
    "day": 1,
    "stop_name": "Senso-ji Temple",
    "coordinates": {"lat": 35.7148, "lng": 139.7967},
    "start_time": "15:30",
    "end_time": "17:30",
    "cost": 0.0,
    "opening_hours": {"open": "06:00", "close": "17:00"},
    "stop_type": "activity"
  },
  {
    "day": 1,
    "stop_name": "Shibuya Ramen Dinner",
    "coordinates": {"lat": 35.6595, "lng": 139.7004},
    "start_time": "18:00",
    "end_time": "20:00",
    "cost": 25.0,
    "opening_hours": {"open": "11:00", "close": "23:00"},
    "stop_type": "activity"
  }
]
</json_itinerary>
"""

report = run_deterministic_verifier(
    llm_itinerary_output=mock_llm_output,
    trip_constraints={"budget": "$1000"},
    user_query="2 day trip to Tokyo under $1000"
)

print("Verification Report:")
import json
print(json.dumps(report, indent=2))
