import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

test_models = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
]

print(f"Testing Groq models with key {api_key[:12]}...")
for m in test_models:
    try:
        llm = ChatGroq(model=m, temperature=0, api_key=api_key)
        res = llm.invoke([HumanMessage(content="Hello! Are you active?")])
        print(f"✅ SUCCESS model='{m}': {res.content.strip()[:100]}")
    except Exception as exc:
        print(f"❌ FAILED model='{m}': {type(exc).__name__} -> {exc}")
