# Weather Advisory Bot

A chatbot that answers outdoor-activity safety questions ("is it safe to cycle today?", "is it good for a picnic?") using **live weather data** from [Open-Meteo](https://open-meteo.com/) matched against written policy rules called SOPs.

Built as a LangGraph agent with a FastAPI backend and a React frontend.

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)

### 1. Clone and create `.env`

```bash
git clone <repo-url>
cd Weather-Advisory-Bot
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

---

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

**Health check:** `GET http://localhost:8000/health`

**API docs:** `http://localhost:8000/docs`

---

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Running the Eval Suite

```bash
cd backend
python evals.py
```

This runs all 8 test cases and prints `PASS`/`FAIL` for each with explanatory notes.

---

## API Reference

### `POST /chat`

**Request:**
```json
{
  "thread_id": "uuid-string",
  "message": "Is it safe to cycle in London today?"
}
```

**Response:**
```json
{
  "reply": "Wind speeds are currently 62.0 km/h...",
  "sop_id": "SOP-001",
  "weather_used": {
    "wind_speed_10m": 62.0,
    "wind_gusts_10m": 78.5
  }
}
```

- `sop_id`: The matched policy rule ID, or `null` if no rule matched or a failure occurred.
- `weather_used`: The exact weather fields substituted into the reply — directly diffable against the Open-Meteo API response for that location. Only contains fields cited in the reply, not the full payload.

---

## Architecture Decisions (Defensible Out Loud)

### Why SOPs are stored as YAML

SOPs are stored in `sops.yaml` and loaded at runtime by `PolicyStore`. No SOP content, category name, threshold value, or field name is hardcoded in Python. This means adding SOP #11 requires editing only the YAML file — zero code changes. During the live review call, you can open `sops.yaml`, add a new entry, and the system will pick it up on the next server restart (or a `PolicyStore` reload) with no code changes.

### Conflict resolution rule

When multiple SOPs match the same query, the system picks the **highest-severity SOP** as the primary response. If a second SOP with a different, non-trivial concern also matched, one short sentence naming it as a secondary consideration is appended — it is not silently dropped, but it is not given equal weight either.

**Why this rule:** Safety advice should lead with the most serious concern. Burying a critical wind warning because a UV concern also triggered would be dangerous. At the same time, silently dropping the secondary concern loses information the user needs. The one-sentence secondary note is a deliberate compromise: it alerts the user without creating conflicting advice or equal-weight confusion.

### Numeric grounding enforcement

The reply-composition LLM call must use `{{field_name}}` placeholders (e.g. `{{wind_speed_10m}}`) rather than writing numbers directly. Python code substitutes the actual fetched values before the reply is returned. If a placeholder references a field not in the fetched weather data, the system logs it as a bug and returns a safe fallback message rather than showing a broken `{{...}}` to the user. The `weather_used` field in the API response contains exactly the placeholder-fed fields — useful for diffing reply numbers in the eval suite.

### resolve_location determinism

`resolve_location` is mostly deterministic:
1. **Regex-first** (pure code): patterns like `"in London"`, `"at Mumbai"` are extracted without LLM involvement.
2. **LLM assist fallback** (only if regex fails): the LLM proposes a place name string — it cannot invent a geocoded location because Open-Meteo geocoding still validates it. An LLM-proposed name that geocodes to nothing routes to `honest_failure`.
3. **Session reuse**: if a location is already in state and no new place is mentioned, both steps are skipped.

---

## Eval Suite Notes

- **Cases 1 & 2** (clear-match): weather data is mocked to force specific thresholds. Numbers in the reply are asserted to match the mocked values via `weather_used`.
- **Cases 3 & 4** (paraphrase): same mocked conditions, reworded questions — testing LLM semantic matching, not string matching.
- **Case 5** (live severe): calls the real Open-Meteo API with no mocking. **Expected values are not hardcoded** — the test asserts on structure (numbers in reply match `weather_used`, SOP selected is valid). The result will vary by actual conditions on the day you run it.
- **Case 6** (no guidance): indoor activity with benign weather — asserts the system honestly admits it has no guidance rather than inventing advice.
- **Case 7** (API failure): `fetch_weather` is monkeypatched to raise `WeatherFetchError`. Asserts the `honest_failure` node fires with a templated message and no fabricated forecast numbers.
- **Case 8** (adversarial): prompt injection attempt. Asserts `SOP-099` is never returned, the reply never says "always safe," and the system either applies a real matched SOP or admits no guidance applies.
