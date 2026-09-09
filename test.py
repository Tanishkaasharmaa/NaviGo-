from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent
from verification.analysis_formatter import print_analysis_report

user_input = input("Enter travel request: ")
if not user_input.strip():
    user_input = "Plan a 3 days trip to Paris under $1000"

print(f"\nRunning NaviGo Travel Agent test for query: '{user_input}'...\n")

response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user",
    stream_realtime=True
)

# Print full analytical report (Test details, LLM outputs, constraint checks, benchmark metrics)
print_analysis_report(response, query=user_input)