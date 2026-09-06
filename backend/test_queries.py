from graph import build_graph, BotState

app = build_graph()

def test_query(msg: str):
    print("=" * 70)
    print(f"QUERY: {msg}")
    initial_state = {
        "thread_id": "test_1",
        "user_message": msg,
        "last_location": None,
        "last_weather": None,
        "last_weather_ts": None,
        "last_sop_id": None,
        "last_user_query": None,
        "numeric_candidate_ids": [],
        "selected_sop_id": None,
        "secondary_sop_id": None,
        "reasoning": "",
        "reply": "",
        "weather_facts": None,
        "failure_reason": None,
    }
    result = app.invoke(initial_state)
    print("RESOLVED LOCATION:", result.get("last_location"))
    print("FAILURE REASON:", result.get("failure_reason"))
    print("SELECTED SOP ID:", result.get("selected_sop_id"))
    print("REASONING:", result.get("reasoning"))
    print("REPLY:\n", result.get("reply"))

if __name__ == "__main__":
    test_query("Is it safe to drive to the mountains today from Berlin?")
    print("\n")
    test_query("Is it safe to cycle in London today?")
    print("\n")
    test_query("Can I go bouldering indoors in London today?")
