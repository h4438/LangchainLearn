from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

def get_openai_model() -> OpenAI:
    load_dotenv()
    ai_pass = os.getenv("PASSPORT")
    os.environ['OPENAI_API_KEY'] = ai_pass
    model = OpenAI(verbose=True, temperature=0.0)
    print("OPENAI ready")
    return model

def enable_tracing(session:str='test-deploy') -> bool:
    load_dotenv()
    lang_key = os.getenv("TRACEPORT")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
    os.environ["LANGCHAIN_API_KEY"] = lang_key
    os.environ["LANGCHAIN_SESSION"] = session
    print(f"Enable tracing at {session}")
    return True

