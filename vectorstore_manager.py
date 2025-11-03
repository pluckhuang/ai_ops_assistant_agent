import os
from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_factory import get_embeddings


class VectorStoreManager:
    def __init__(self, data_path="data/sample.txt", base_path="faiss_index"):
        self.data_path = data_path
        self.base_path = Path(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    def get_index_path(self, mode="local"):
        return self.base_path / mode

    def build_vectorstore(self, mode="local", force_rebuild=False):
        index_path = self.get_index_path(mode)
        if index_path.exists() and not force_rebuild:
            embeddings = get_embeddings(mode)
            return FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )

        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = [
            Document(page_content=chunk) for chunk in self.splitter.split_text(text)
        ]
        embeddings = get_embeddings(mode)
        vectorstore = FAISS.from_documents(docs, embeddings)

        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        print(f"✅ 已构建 {mode} 向量库，路径: {index_path}")
        return vectorstore
