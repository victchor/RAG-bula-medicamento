from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Modelo multilíngue leve, treinado em pares (query, passagem) — roda
# local, sem custo de API.
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(reranker: CrossEncoder, pergunta: str, documentos: list[Document], k_final: int) -> list[Document]:
    if not documentos:
        return []

    # Cross-encoder: pergunta e chunk entram juntos no modelo (atenção
    # cruzada), diferente do embedding, que processa cada um separadamente.
    # Mais lento por isso — só é viável aqui porque já reduzimos a lista
    # aos candidatos vindos da busca vetorial.
    pares = [(pergunta, doc.page_content) for doc in documentos]
    scores = reranker.predict(pares)

    documentos_por_score = sorted(zip(scores, documentos), key=lambda par: par[0], reverse=True)
    return [doc for _, doc in documentos_por_score[:k_final]]
