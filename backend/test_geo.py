from weather import geocode, LocationNotFoundError
from graph import _extract_location_regex, _extract_location_llm

test_inputs = [
    "Hello can i go for a night ride in bangaluru",
    "Hello can i go for a night ride in Hyderabad",
    "Hyderabad",
    "bengaluru",
]

for text in test_inputs:
    cand_reg = _extract_location_regex(text)
    print(f"Text: {text!r} -> Regex candidate: {cand_reg!r}")
    try:
        if cand_reg:
            res = geocode(cand_reg)
            print(f"  Geocoded regex: {res}")
    except LocationNotFoundError as e:
        print(f"  Geocoded regex FAILED: {e}")

    cand_llm = _extract_location_llm(text)
    print(f"  LLM candidate: {cand_llm!r}")
    try:
        if cand_llm:
            res = geocode(cand_llm)
            print(f"  Geocoded LLM: {res}")
    except LocationNotFoundError as e:
        print(f"  Geocoded LLM FAILED: {e}")
