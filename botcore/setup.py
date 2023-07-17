from langchain.llms import OpenAI, AI21
from langchain.chat_models import ChatOpenAI, PromptLayerChatOpenAI
import os
from dotenv import load_dotenv

def load_my_env():
    env_path = os.path.dirname(__file__)
    load_dotenv(f'{env_path}/.env')

## TRACE
def trace_chat_openai(model_name: str = 'text-davinci-003' ,max_tokens: int = 256, session:str='test-deploy') -> ChatOpenAI:
    enable_tracing()
    ai_pass = os.getenv("OPENAI")
    os.environ['OPENAI_API_KEY'] = ai_pass
    model = ChatOpenAI(model_name=model_name, max_tokens=max_tokens,verbose=True, temperature=0.0)
    print("CHAT OPENAI ready")
    return model

def trace_openai(model_name: str = 'text-davinci-003' ,max_tokens: int = 256, session: str = 'test-deploy') -> OpenAI:
    enable_tracing()
    ai_pass = os.getenv("OPENAI")
    os.environ['OPENAI_API_KEY'] = ai_pass
    model = OpenAI(model_name=model_name, max_tokens=max_tokens,verbose=True, temperature=0.0)
    print("OPENAI ready")
    return model

def trace_ai21(model_name: str = 'j2-jumbo-instruct', max_tokens: int = 256, session: str = 'test-deploy') -> AI21:
    enable_tracing()
    ai_pass = os.getenv("AI21")
    model = AI21(ai21_api_key=ai_pass, model=model_name, maxTokens=max_tokens, temperature=0.0)
    print("AI21 ready")
    return model

## TRACING

def enable_tracing(session:str='test-deploy') -> bool:
    load_my_env()
    lang_key = os.getenv("LANGCHAIN")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
    os.environ["LANGCHAIN_API_KEY"] = lang_key
    os.environ["LANGCHAIN_SESSION"] = session
    print(f"Enable tracing at {session}")
    return True
