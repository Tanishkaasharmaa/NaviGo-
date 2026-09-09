import asyncio
import unittest
from verification.existence_gate import resolve_place
from verification.constraints import (
    check_budget,
    check_travel_time,
    check_hotel_dates,
    check_daily_hours,
    run_all_checks,
)
from benchmark.adapter import evaluate_itinerary_result


class TestNaviGoNewFeatures(unittest.TestCase):

    def test_existence_gate_osm_resolution(self):
        """Test resolving an iconic real place using OpenStreetMap fallback."""
        res = asyncio.run(resolve_place("Eiffel Tower", city_context="Paris"))
        self.assertIsNotNone(res)
        self.assertTrue(res.get("verified"))
        self.assertIsNotNone(res.get("lat"))
        self.assertIsNotNone(res.get("lon"))
        self.assertIn("Paris", res.get("resolved_name", "") + str(res.get("source")))

    def test_existence_gate_invalid_place(self):
        """Test resolving a hallucinated/nonexistent attraction."""
        res = asyncio.run(resolve_place("Sacred Canyon of Humantay Nonexistent 9999", city_context="Remote Andes"))
        self.assertIsNotNone(res)
        self.assertFalse(res.get("verified"))
        self.assertIsNotNone(res.get("unverified_reason"))

    def test_budget_constraint_verifier(self):
        """Test deterministic budget constraint check."""
        itinerary = {"total_cost": 1500}
        violations = check_budget(itinerary, stated_budget="$1000")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["category"], "budget")
        self.assertIn("1500.00", violations[0]["description"])

    def test_hotel_dates_verifier(self):
        """Test hotel check-in/out dates verifier."""
        itinerary = {
            "hotel": {"name": "Grand Hotel", "check_in_day": 2, "check_out_day": 3},
            "days": [{"day": 1}, {"day": 2}, {"day": 3}, {"day": 4}]
        }
        violations = check_hotel_dates(itinerary, total_days=4)
        self.assertGreaterEqual(len(violations), 1)
        categories = [v["category"] for v in violations]
        self.assertIn("hotel_dates", categories)

    def test_daily_activity_ceiling_verifier(self):
        """Test daily activity ceiling (> 10 hours)."""
        days = [
            {
                "day": 1,
                "stops": [
                    {"name": "Stop A", "start_time": "07:00", "end_time": "12:00"},
                    {"name": "Stop B", "start_time": "13:00", "end_time": "22:00"},
                ]
            }
        ]
        violations = check_daily_hours(days, ceiling_hours=10.0)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["category"], "daily_hours")

    def test_travel_time_consecutive_buffer_verifier(self):
        """Test travel time check between consecutive stops."""
        days = [
            {
                "day": 1,
                "stops": [
                    {
                        "name": "Eiffel Tower",
                        "start_time": "09:00",
                        "end_time": "11:55",
                        "lat": 48.8584,
                        "lon": 2.2945,
                        "verified": True
                    },
                    {
                        "name": "Louvre Museum",
                        "start_time": "12:00",
                        "end_time": "15:00",
                        "lat": 48.8606,
                        "lon": 2.3376,
                        "verified": True
                    }
                ]
            }
        ]
        violations = asyncio.run(check_travel_time(days))
        # 5 minutes gap (11:55 to 12:00) between Eiffel Tower and Louvre (approx 3.5km) should trigger buffer warning
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["category"], "travel_time")

    def test_run_all_checks_score(self):
        """Test constraint satisfaction score calculation."""
        itinerary = {
            "total_cost": 500,
            "days": [{"day": 1, "stops": []}],
            "hotel": {"name": "Valid Hotel", "check_in_day": 1, "check_out_day": 1}
        }
        constraints = {"budget": "$1000"}
        report = asyncio.run(run_all_checks(itinerary, constraints))
        self.assertEqual(report["score"], 1.0)
        self.assertEqual(len(report["violations"]), 0)

    def test_benchmark_adapter_evaluation(self):
        """Test TravelPlanner benchmark metric adapter."""
        sample = {"query": "Plan 3 days to Paris under $500."}
        result = {
            "answer": "Here is a complete narrative travel plan for Paris...",
            "itinerary_json": {
                "days": [{"day": 1, "stops": [{"name": "Stop 1"}]}]
            },
            "verification_report": {"score": 1.0, "violations": []}
        }
        eval_res = evaluate_itinerary_result(sample, result)
        self.assertTrue(eval_res["delivered"])
        self.assertTrue(eval_res["final_pass"])
        self.assertEqual(eval_res["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
