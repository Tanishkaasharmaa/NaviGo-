import os
import certifi
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


import re
from datetime import datetime
from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)
from verification.existence_gate import resolve_place
from verification.constraints import run_all_checks


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# =========================
# LLM
# =========================
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    max_tokens=2048,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # Structured verification + provenance state
    itinerary_json: dict[str, Any]
    itinerary_retry_count: int
    verification_report: dict[str, Any]
    sources: list[dict[str, Any]]

    # Budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


import time


def _llm_invoke_with_retry(messages: list[AnyMessage], max_retries: int = 3) -> Any:
    """Invoke LLM with automatic retry on Groq rate limits (HTTP 429)."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc).lower()
            if "rate limit" in err_str or "429" in err_str:
                sleep_time = (attempt + 1) * 7
                print(f"[LLM] Groq Rate Limit (429). Waiting {sleep_time}s before retry ({attempt + 1}/{max_retries})...", flush=True)
                time.sleep(sleep_time)
            else:
                raise exc
    return llm.invoke(messages)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = _llm_invoke_with_retry(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _extract_fenced_json(text: str) -> dict[str, Any]:
    """Extract fenced ```json ... ``` block or fall back to brace matching."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    try:
        return _json_from_llm(text)
    except Exception:
        return {"days": [], "hotel": {}, "total_cost": 0}


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
        "trip_start_date": "",
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": [],
    "trip_start_date": ""
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - original behavior kept
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:1000],
            airline_data=str(airlines)[:1000],
        )

        response = _llm_invoke_with_retry(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    current_sources = list(state.get("sources", []))

    try:
        hotel_results = asyncio.run(tavily_mcp_search(query))
        raw_str = str(hotel_results)
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', raw_str)
        for u in urls[:3]:
            if not any(s.get("url") == u for s in current_sources):
                current_sources.append({
                    "title": f"Hotel Search: {query}",
                    "url": u,
                    "fetched_at": datetime.utcnow().isoformat(),
                    "used_by": "hotel_agent"
                })

    except Exception as exc:
        print(f"HOTEL AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        hotel_results = (
            "Live hotel search is temporarily unavailable. Provide general accommodation "
            "and neighborhood guidance based on the destination."
        )

    return {
        "hotel_results": str(hotel_results),
        "sources": current_sources,
        "messages": [AIMessage(content="Hotel information processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(state["user_query"])

    try:
        weather_data = asyncio.run(weather_mcp_search(city))
        forecast_data = asyncio.run(forecast_mcp_search(city))

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""
    except Exception as exc:
        print(f"WEATHER AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        weather_results = (
            f"Live weather information for {city} is temporarily unavailable. "
            "Give general seasonal guidance."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{str(state.get('flight_results', ''))[:800]}

Hotel Results:
{str(state.get('hotel_results', ''))[:800]}

Weather Results:
{str(state.get('weather_results', ''))[:400]}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = _llm_invoke_with_retry(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - structured JSON output + violation feedback handling
# =========================
def itinerary_agent(state: TravelState):
    retry_count = state.get("itinerary_retry_count", 0)
    report = state.get("verification_report", {})
    violations = report.get("violations", [])

    violation_feedback = ""
    if retry_count > 0 and violations:
        violation_text = "\n".join([f"- {v.get('description')}" for v in violations])
        violation_feedback = f"""
IMPORTANT: Your previous itinerary draft produced these specific constraint violations:
{violation_text}

Please revise the itinerary to fix ONLY these violations while preserving everything else that was valid.
"""

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{str(state.get('flight_results', ''))[:800]}

Hotel Results:
{str(state.get('hotel_results', ''))[:800]}

Weather Results:
{str(state.get('weather_results', ''))[:400]}

Budget Results:
{str(state.get('budget_results', ''))[:800]}
{violation_feedback}

Make the itinerary practical, budget-aware, and easy to follow.
After your narrative response, output exactly one fenced JSON block containing the structured itinerary schema:

```json
{{
  "days": [
    {{
      "day": 1,
      "stops": [
        {{
          "name": "Attraction Name",
          "address": "Attraction Address or Neighborhood",
          "start_time": "09:00",
          "end_time": "11:30",
          "cost": 25,
          "category": "attraction"
        }}
      ]
    }}
  ],
  "hotel": {{
    "name": "Hotel Name",
    "check_in_day": 1,
    "check_out_day": 3,
    "total_cost": 300
  }},
  "total_cost": 450
}}
```
Do not include lat/lon coordinates in the JSON — those will be resolved separately.
"""

    response = _llm_invoke_with_retry(
        [
            SystemMessage(content="You are an expert travel planner. Always end with a fenced JSON itinerary block."),
            HumanMessage(content=prompt),
        ]
    )

    itinerary_text = str(response.content)
    itinerary_json = _extract_fenced_json(itinerary_text)

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": itinerary_text,
        "itinerary_json": itinerary_json,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created with structured JSON.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


MAX_ITINERARY_RETRIES = 2


# =========================
# Existence Gate Agent
# =========================
def existence_gate_agent(state: TravelState):
    itinerary_json = dict(state.get("itinerary_json", {}))
    destination = state.get("trip_constraints", {}).get("destination", "")
    if not destination:
        try:
            destination = extract_destination(state["user_query"])
        except Exception:
            destination = ""

    sources = list(state.get("sources", []))
    days = itinerary_json.get("days", [])

    for day in days:
        stops = day.get("stops", [])
        for stop in stops:
            name = stop.get("name", "")
            if name:
                res = asyncio.run(resolve_place(name, city_context=destination))
                stop["lat"] = res.get("lat")
                stop["lon"] = res.get("lon")
                stop["verified"] = res.get("verified", False)
                stop["place_id"] = res.get("place_id")
                stop["unverified_reason"] = res.get("unverified_reason")
                if res.get("source"):
                    if not any(s.get("url") == res["source"].get("url") for s in sources):
                        sources.append(res["source"])

    hotel = itinerary_json.get("hotel")
    if hotel and isinstance(hotel, dict) and hotel.get("name"):
        res = asyncio.run(resolve_place(hotel["name"], city_context=destination))
        hotel["lat"] = res.get("lat")
        hotel["lon"] = res.get("lon")
        hotel["verified"] = res.get("verified", False)
        hotel["place_id"] = res.get("place_id")
        hotel["unverified_reason"] = res.get("unverified_reason")
        if res.get("source"):
            if not any(s.get("url") == res["source"].get("url") for s in sources):
                sources.append(res["source"])

    return {
        "itinerary_json": itinerary_json,
        "sources": sources,
        "messages": [AIMessage(content="Existence gate verification completed.")],
    }


# =========================
# Verifier Agent
# =========================
def verifier_agent(state: TravelState):
    itinerary_json = state.get("itinerary_json", {})
    constraints = state.get("trip_constraints", {})

    report = asyncio.run(run_all_checks(itinerary_json, constraints))

    return {
        "verification_report": report,
        "messages": [AIMessage(content=f"Verifier executed. Score: {report.get('score')}")]
    }


def route_after_verifier(state: TravelState) -> str:
    report = state.get("verification_report", {})
    retries = state.get("itinerary_retry_count", 0)
    violations = report.get("violations", [])

    if violations and retries < MAX_ITINERARY_RETRIES:
        print(f"[Verifier] Violations found ({len(violations)}). Attempt {retries + 1}/{MAX_ITINERARY_RETRIES}. Retrying itinerary_agent...")
        return "retry_itinerary"

    return "human_approval"


def increment_retry_agent(state: TravelState):
    return {
        "itinerary_retry_count": state.get("itinerary_retry_count", 0) + 1
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "verification_report": state.get("verification_report", {}),
            "sources": state.get("sources", []),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    report = state.get("verification_report", {})
    violations_summary = ""
    if report.get("violations"):
        v_strs = [f"- {v['description']}" for v in report["violations"]]
        violations_summary = f"\nVerification Warnings:\n" + "\n".join(v_strs)

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}
{violations_summary}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = _llm_invoke_with_retry(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("existence_gate_agent", existence_gate_agent)
graph.add_node("verifier_agent", verifier_agent)
graph.add_node("increment_retry_agent", increment_retry_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "existence_gate_agent")
graph.add_edge("existence_gate_agent", "verifier_agent")
graph.add_conditional_edges(
    "verifier_agent",
    route_after_verifier,
    {
        "retry_itinerary": "increment_retry_agent",
        "human_approval": "human_approval",
    }
)
graph.add_edge("increment_retry_agent", "itinerary_agent")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "itinerary_json": result.get("itinerary_json", {}),
        "verification_report": (
            interrupt_payload.get("verification_report", {})
            if interrupt_payload
            else result.get("verification_report", {})
        ),
        "sources": (
            interrupt_payload.get("sources", [])
            if interrupt_payload
            else result.get("sources", [])
        ),
        "itinerary_retry_count": result.get("itinerary_retry_count", 0),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None, stream_realtime: bool = False):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": _empty_constraints(),
        "supervisor_reasoning": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "itinerary_json": {},
        "itinerary_retry_count": 0,
        "verification_report": {},
        "sources": [],
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": 0,
    }

    if stream_realtime:
        from verification.analysis_formatter import print_realtime_node_update

        print("\n" + "=" * 80)
        print(" ⚡ REAL-TIME NAVIGO GRAPH EXECUTION STREAM ⚡")
        print("=" * 80)
        for event in travel_graph.stream(initial_input, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                print_realtime_node_update(node_name, update)
        print("\n" + "=" * 80 + "\n")
        snapshot = travel_graph.get_state(config)
        return _serialize_result(snapshot.values, thread_id)

    result = travel_graph.invoke(initial_input, config=config)

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)


def benchmark_travel_agent(user_input: str, verifier_enabled: bool = True):
    """Non-interactive graph execution wrapper for automated benchmarking."""
    thread_id = f"bench_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    if not verifier_enabled:
        # Run standard flow bypassing retry loop
        result = travel_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "guardrail_allowed": True,
                "guardrail_reason": "",
                "selected_agents": [],
                "trip_constraints": _empty_constraints(),
                "supervisor_reasoning": "",
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "budget_results": "",
                "itinerary": "",
                "itinerary_json": {},
                "itinerary_retry_count": MAX_ITINERARY_RETRIES,  # Skip retry loop
                "verification_report": {},
                "sources": [],
                "approval_request": "",
                "approved": True,
                "human_feedback": "",
                "final_response": "",
                "llm_calls": 0,
            },
            config=config,
        )
    else:
        result = travel_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "guardrail_allowed": True,
                "guardrail_reason": "",
                "selected_agents": [],
                "trip_constraints": _empty_constraints(),
                "supervisor_reasoning": "",
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "budget_results": "",
                "itinerary": "",
                "itinerary_json": {},
                "itinerary_retry_count": 0,
                "verification_report": {},
                "sources": [],
                "approval_request": "",
                "approved": True,
                "human_feedback": "",
                "final_response": "",
                "llm_calls": 0,
            },
            config=config,
        )

    # Auto resume if interrupted by HITL
    if _interrupt_payload(result):
        result = travel_graph.invoke(Command(resume={"approved": True, "feedback": ""}), config=config)

    return _serialize_result(result, thread_id)