from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI


def get_llm(provider="ollama", model_name="llama3.2:3b", temperature=0.2):
    """Get LLM instance based on provider"""
    if provider.lower() == "ollama":
        return OllamaLLM(model=model_name, temperature=temperature)
    elif provider.lower() == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
