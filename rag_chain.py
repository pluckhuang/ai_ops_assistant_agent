from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llm_factory import get_llm
from vectorstore_manager import VectorStoreManager


def format_docs(docs):
    """格式化文档为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


def load_qa_chain(
    provider="ollama",
    model_name="llama3.2:3b",
    temperature=0.2,
    embedding_mode="ollama",
):
    """创建基于检索的问答链

    返回一个可调用的 chain 对象，使用方式:
    result = qa_chain.invoke({"question": "你的问题"})
    """

    # 初始化向量存储和检索器
    vsm = VectorStoreManager()
    vectorstore = vsm.build_vectorstore(mode=embedding_mode)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 获取 LLM
    llm = get_llm(provider=provider, model_name=model_name, temperature=temperature)

    # 创建 RAG 提示模板
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in a helpful and concise way:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 创建 RAG 链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
