import sys

from langchain_core.messages import AIMessage, HumanMessage

from app.config import VECTORSTORE_DIR, load_settings
from app.ingestion.embeddings import get_embeddings
from app.ingestion.vectorstore import load_existing_vectorstore
from app.llm import get_chat_model
from app.rag.chain import get_rag_chain
from app.rag.reranker import get_reranker
from app.rag.retriever import get_retriever

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    settings = load_settings()
    llm = get_chat_model(settings)
    embeddings = get_embeddings()

    vectorstore = load_existing_vectorstore(
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    retriever = get_retriever(vectorstore)
    reranker = get_reranker()
    chain = get_rag_chain(retriever, reranker, llm)

    # A 2a pergunta só faz sentido em conjunto com a 1a ("E quais são"
    # não tem sujeito sozinha) — testa a reformulação de pergunta.
    perguntas = [
        "O que é este medicamento?",
        "E quais são os efeitos colaterais?",
    ]

    chat_history: list[HumanMessage | AIMessage] = []

    for pergunta in perguntas:
        print("=" * 70)
        print(f"PERGUNTA: {pergunta}")
        print(f"(histórico até aqui: {len(chat_history)} mensagens)\n")

        resultado = chain.invoke({"question": pergunta, "chat_history": chat_history})
        resposta = resultado["answer"]
        print(f"RESPOSTA: {resposta}\n")
        print(f"Fontes: {[d.metadata.get('page_label') for d in resultado['source_documents']]}\n")

        chat_history.append(HumanMessage(content=pergunta))
        chat_history.append(AIMessage(content=resposta))


if __name__ == "__main__":
    main()
