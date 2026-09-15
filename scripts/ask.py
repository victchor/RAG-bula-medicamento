import sys
import time

from langchain_core.messages import AIMessage, HumanMessage

from app.config import VECTORSTORE_DIR, load_settings
from app.ingestion.embeddings import get_embeddings
from app.ingestion.vectorstore import load_existing_vectorstore
from app.llm import get_chat_model
from app.observability.handler import LiveLogHandler, ObservabilityHandler, formatar_metricas
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
    nome_modelo = settings.claude_model if settings.llm_provider == "anthropic" else settings.gemini_model

    print("Chat com a bula indexada. Digite 'sair' para encerrar.\n")

    chat_history: list[HumanMessage | AIMessage] = []

    while True:
        pergunta = input("Você: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        observador = ObservabilityHandler()
        inicio = time.perf_counter()
        print()
        resultado = chain.invoke(
            {"question": pergunta, "chat_history": chat_history},
            config={"callbacks": [observador, LiveLogHandler()]},
        )
        duracao = time.perf_counter() - inicio
        resposta = resultado["answer"]

        paginas_finais = [d.metadata.get("page_label") for d in resultado["source_documents"]]

        print(f"\nBot: {resposta}\n")
        print(f"[{formatar_metricas(observador.metricas, duracao, nome_modelo, paginas_finais)}]\n")

        chat_history.append(HumanMessage(content=pergunta))
        chat_history.append(AIMessage(content=resposta))


if __name__ == "__main__":
    main()
