# ✈️ NaviGo — Your AI Travel Planner with LangGraph, MCP & Deterministic Verification

An open-source AI travel planner that turns a natural-language trip request into a practical travel plan with flight suggestions, hotel ideas, weather forecasts, budget analysis, and a day-by-day itinerary. The project features a multi-agent workflow built with **LangGraph**, **LangChain**, **FastAPI**, **Model Context Protocol (MCP)**, and a **Deterministic Hard-Constraint Verifier**.

---

## 🌟 Why this project?

Planning a trip usually means jumping between multiple websites, tools, and spreadsheets. NaviGo brings that flow into one experience by combining specialist agents coordinated through a **LangGraph state graph**:

- 🛡️ **Input Guardrail Agent** (Validates travel request scope)
- 🎯 **Supervisor Router Agent** (Extracts constraints & routes to specialists)
- ✈️ **Flight Research Agent** (AviationStack API / MCP)
- 🏨 **Hotel Research Agent** (Tavily Search API / MCP)
- 🌦️ **Weather Forecast Agent** (FastMCP OpenWeather Server)
- 💰 **Budget Feasibility Agent** (Financial analysis)
- 🗓️ **Itinerary Planning Agent** (Markdown plan + structured `<json_itinerary>` stops)
- 🛡️ **Deterministic Verifier Agent** (Pure Python 5-point constraint checker & score engine)
- 👤 **Human-in-the-Loop Agent** (Interactive review & feedback interrupt)
- 📝 **Final Synthesis Agent** (Polished response & PDF export)

---

## ⚡ Features

- ✈️ **Flight Research**: Live airport schedules and airline routes using AviationStack.
- 🏨 **Hotel Suggestions**: Live accommodation research using Tavily Search MCP.
- 🌦️ **Weather Forecasts**: Live weather conditions and 5-period forecast using a custom FastMCP server.
- 🧠 **Multi-Agent Orchestration**: Dynamic routing and state persistence with LangGraph & PostgreSQL.
- 🛡️ **Deterministic Itinerary Verifier**: Pure Python validation engine (zero LLM hallucinations) checking 5 hard constraints:
  1. **Budget Limit**: Total cost vs stated budget constraint.
  2. **OSRM Transit Time**: Real travel time between coordinates via OSRM API (with Haversine fallback).
  3. **Opening Hours**: Time slot alignment with venue operating hours.
  4. **Hotel Check-in**: Arrival day (Day 1) check-in & overlap prevention.
  5. **Daily Activity Ceiling**: Maximum 10-hour daily active schedule limit.
- 📊 **Constraint Satisfaction Score**: Surfaces a 0–100% verification scorecard in the web UI.
- 👤 **Human-in-the-Loop (HITL)**: Native LangGraph `interrupt()` node allowing travelers to approve or revise draft itineraries before finalizing.
- 💾 **State Persistence**: PostgreSQL checkpointer (`PostgresSaver`) for multi-turn session state.
- 🌐 **FastAPI & Modern Web UI**: Responsive dark glassmorphic UI with PDF export capability (`html2pdf.js`).

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastAPI & Uvicorn**
- **Jinja2 + HTML5 / CSS3 / JavaScript**
- **LangGraph & LangChain**
- **Groq LLMs** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
- **Model Context Protocol (MCP)** (`mcp`, `langchain-mcp-adapters`, `FastMCP`)
- **PostgreSQL** (`psycopg`, `langgraph-checkpoint-postgres`)
- **OSRM API** (Open Source Routing Machine)
- **Tavily API & AviationStack API & OpenWeather API**

---

## 📁 Project Structure

```text
.
├── app.py                         # FastAPI application & REST endpoints
├── backend.py                     # LangGraph travel workflow & agent nodes
├── mcp_client.py                  # MultiServerMCPClient configuration
├── custom_weather_mcp_server.py   # FastMCP Weather Server (stdio transport)
├── requirements.txt               # Python dependencies
├── static/                        # CSS styles & client-side JavaScript
├── templates/                     # HTML templates (Jinja2)
├── tools/                         # Flight and Tavily search helper scripts
└── verification/                  # Deterministic verifier engine & OSRM routing
    └── verifier.py                # Pure Python hard-constraint checker
```

---

## 📋 Prerequisites

Before running the project locally, make sure you have:

- **Python 3.10 or newer** installed
- **PostgreSQL** running and accessible
- API keys for:
  - Groq
  - Tavily
  - AviationStack
  - OpenWeather

---

## 🔑 Environment Variables

Create a `.env` file in the project root with the following variables:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/travel_db
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
DEFAULT_ORIGIN_IATA=DAC
```

---

## 📦 Installation

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the App

Start the FastAPI server:

```bash
python app.py
```

Then open your browser at:

```text
http://127.0.0.1:8000/
```

---

## 🔌 API Endpoints

- `GET /health` — API health & feature status check
- `POST /api/travel` — Submit a travel request (runs graph until human approval interrupt)
- `POST /api/travel/approve` — Resume workflow with human approval or revision feedback

### Example Travel Request:

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan a 3-day trip to Tokyo from Bangladesh under $1200"}'
```

---

## 🔄 How the Workflow Works

1. **Request Submission**: The user submits a travel request.
2. **Guardrail Check**: The Input Guardrail verifies that the query belongs to the travel domain.
3. **Supervisor Routing**: The Supervisor Agent extracts trip constraints and dispatches work to necessary specialists.
4. **Specialist Research**: Flight, Hotel, Weather, and Budget agents gather live data via MCP servers and APIs.
5. **Draft Generation**: The Itinerary Agent generates markdown prose and structured JSON stop coordinates.
6. **Deterministic Verification**: The Verifier Agent evaluates 5 hard constraints (Budget, OSRM Travel Gaps, Opening Hours, Hotel Check-in, Activity Ceiling). If violations are found, it loops back to the Itinerary Agent with correction feedback (capped at 2 retries).
7. **Human Review**: The workflow pauses via `interrupt()`, displaying the draft plan and **Constraint Satisfaction Score** on the web UI.
8. **Final Synthesis**: Upon human approval or revision feedback, the Final Response Agent creates the polished travel plan ready for export.

---

## 🤝 Contributing

Contributions are welcome! If you want to improve the app, add new travel features, or fix issues:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.