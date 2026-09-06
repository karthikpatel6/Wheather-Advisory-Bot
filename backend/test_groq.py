import os
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    models = [m.id for m in client.models.list().data]
    print("AVAILABLE GROQ MODELS:", models)
except Exception as e:
    print("GROQ CLIENT ERROR:", type(e), e)

try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv('GROQ_API_KEY'))
    res = llm.invoke("Hello")
    print("LANGCHAIN GROQ SUCCESS:", res.content)
except Exception as e:
    print("LANGCHAIN GROQ ERROR:", type(e), e)
