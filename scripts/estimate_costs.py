import sys

from langchain_core.messages import AIMessage, HumanMessage

from app.config import VECTORSTORE_DIR, load_settings
from app.ingestion.embeddings import get_embeddings
from app.ingestion.vectorstore import load_existing_vectorstore
from app.llm import get_chat_model
from app.observability.handler import PRECOS_POR_MILHAO_TOKENS, ObservabilityHandler
from app.rag.chain import get_rag_chain
from app.rag.reranker import get_reranker
from app.rag.retriever import get_retriever

sys.stdout.reconfigure(encoding="utf-8")

MODELOS_COMPARADOS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5",
    "claude-opus-5",
    "gemini-3.1-flash-lite-paga",
]


def custo(tokens_entrada: int, tokens_saida: int, modelo: str) -> float:
    precos = PRECOS_POR_MILHAO_TOKENS[modelo]
    return (tokens_entrada / 1_000_000) * precos["entrada"] + (tokens_saida / 1_000_000) * precos["saida"]


def main() -> None:
    settings = load_settings()
    llm = get_chat_model(settings)
    embeddings = get_embeddings()

    vectorstore = load_existing_vectorstore(embedding=embeddings, persist_directory=str(VECTORSTORE_DIR))
    retriever = get_retriever(vectorstore)
    reranker = get_reranker()
    chain = get_rag_chain(retriever, reranker, llm)

    obs_isolada = ObservabilityHandler()
    resultado1 = chain.invoke(
        {"question": "Para que serve este medicamento?", "chat_history": []},
        config={"callbacks": [obs_isolada]},
    )

    historico = [
        HumanMessage(content="Para que serve este medicamento?"),
        AIMessage(content=resultado1["answer"]),
    ]
    obs_acompanhamento = ObservabilityHandler()
    chain.invoke(
        {"question": "E quais são os efeitos colaterais?", "chat_history": historico},
        config={"callbacks": [obs_acompanhamento]},
    )

    print("=" * 70)
    print("TOKENS MEDIDOS (dado real, não estimado)")
    print(
        f"Pergunta isolada:        chamadas_llm={obs_isolada.metricas.chamadas_llm}  "
        f"entrada={obs_isolada.metricas.tokens_entrada}  saida={obs_isolada.metricas.tokens_saida}"
    )
    print(
        f"Pergunta de acompanhamento: chamadas_llm={obs_acompanhamento.metricas.chamadas_llm}  "
        f"entrada={obs_acompanhamento.metricas.tokens_entrada}  saida={obs_acompanhamento.metricas.tokens_saida}"
    )

    print("\n" + "=" * 70)
    print("CUSTO POR PERGUNTA, POR MODELO (USD)")
    custo_medio_por_modelo = {}
    for modelo in MODELOS_COMPARADOS:
        c_isolada = custo(obs_isolada.metricas.tokens_entrada, obs_isolada.metricas.tokens_saida, modelo)
        c_acompanhamento = custo(
            obs_acompanhamento.metricas.tokens_entrada, obs_acompanhamento.metricas.tokens_saida, modelo
        )
        custo_medio_por_modelo[modelo] = (c_isolada + c_acompanhamento) / 2
        print(f"{modelo:32s} isolada=${c_isolada:.6f}   acompanhamento=${c_acompanhamento:.6f}")

    print("\n" + "=" * 70)
    print("PROJEÇÃO MENSAL (metade das perguntas com histórico, metade sem)")
    for perguntas_por_dia in (10, 100, 1000):
        print(f"\n{perguntas_por_dia} perguntas/dia:")
        for modelo, media in custo_medio_por_modelo.items():
            mensal = media * perguntas_por_dia * 30
            print(f"  {modelo:32s} ${mensal:.2f}/mês")


if __name__ == "__main__":
    main()
