from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

# Número de candidatos buscados por similaridade vetorial ANTES do
# reranking — maior que o número final usado no prompt (ver K_FINAL em
# chain.py) de propósito, para dar ao reranker mais opções para escolher.
K_CANDIDATOS = 15


def get_retriever(vectorstore: Chroma, k: int = K_CANDIDATOS) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})
