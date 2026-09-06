import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

try:
    from google import genai
except ImportError:
    genai = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print(f"Key loaded: {gemini_key[:10]}..." if gemini_key else "No key loaded")

if genai and gemini_key:
    try:
        client = genai.Client(api_key=gemini_key)
        print("\n--- Available Google Gemini Models from API ---")
        for model in client.models.list():
            if "generateContent" in getattr(model, "supported_actions", []):
                print(f"  • {model.name}")
    except Exception as exc:
        print(f"Error listing models with google.genai client: {exc}")

test_models = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

def _get_content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""

print("\n--- Testing ChatGoogleGenerativeAI Invocation ---")
for m in test_models:
    try:
        if ChatGoogleGenerativeAI and gemini_key:
            llm = ChatGoogleGenerativeAI(model=m, temperature=0, google_api_key=gemini_key)
            res = llm.invoke([HumanMessage(content="Hello! Are you active?")])
            content = _get_content_text(getattr(res, "content", res))
            print(f"✅ SUCCESS model='{m}': {content.strip()[:100]}")
    except Exception as exc:
        print(f"❌ FAILED model='{m}': {type(exc).__name__} -> {exc}")


