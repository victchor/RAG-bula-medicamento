import sys

from app.config import UPLOADS_DIR, VECTORSTORE_DIR
from app.ingestion.embeddings import get_embeddings
from app.ingestion.vectorstore import load_existing_vectorstore
from app.rag.retriever import get_retriever

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    embeddings = get_embeddings()
    vectorstore = load_existing_vectorstore(
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    retriever = get_retriever(vectorstore, k=4)

    pergunta = "Como devo armazenar este medicamento?"
    chunks_recuperados = retriever.invoke(pergunta)

    print(f"Pergunta: {pergunta}")
    print(f"Chunks recuperados (k=4):\n")
    for i, doc in enumerate(chunks_recuperados, start=1):
        print(f"{i}. página {doc.metadata.get('page_label')}")
        print(f"   {doc.page_content[:150]!r}\n")

    # similarity_search_with_score expõe o score (BaseRetriever não). No
    # Chroma, por padrão, é distância — menor valor = mais similar.
    print("-" * 60)
    print("Mesma busca, agora com o score de similaridade visível:\n")
    resultados_com_score = vectorstore.similarity_search_with_score(pergunta, k=4)
    for doc, score in resultados_com_score:
        print(f"score={score:.4f} | página {doc.metadata.get('page_label')} | {doc.page_content[:80]!r}")


if __name__ == "__main__":
    main()
