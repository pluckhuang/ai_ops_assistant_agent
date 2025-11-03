from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()


def get_embeddings(provider="local"):
    if provider == "local":
        return LocalEmbeddings()
    elif provider == "ollama":
        # 使用 Ollama 嵌入模型
        try:
            return OllamaEmbeddings(
                model="nomic-embed-text:latest"  # 或者 "mxbai-embed-large:latest"
            )
        except Exception as e:
            print(f"⚠️ Ollama 嵌入模型不可用，回退到本地模型: {e}")
            print("💡 提示：运行 'ollama pull nomic-embed-text' 来下载 Ollama 嵌入模型")
            return LocalEmbeddings()
    elif provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"未知嵌入模式: {provider}")
