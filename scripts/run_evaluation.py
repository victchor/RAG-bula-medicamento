import importlib
import sys
from collections import Counter

from app.config import VECTORSTORE_DIR, load_settings
from app.evaluation.metrics import avaliar_resposta_com_llm_juiz, hit_rate
from app.ingestion.embeddings import get_embeddings
from app.ingestion.vectorstore import load_existing_vectorstore
from app.llm import get_chat_model
from app.rag.chain import K_FINAL, get_rag_chain
from app.rag.reranker import get_reranker, rerank
from app.rag.retriever import get_retriever

sys.stdout.reconfigure(encoding="utf-8")


def carregar_dataset(nome: str) -> list:
    # Import dinâmico de app/evaluation/datasets/<nome>.py — permite rodar
    # `python -m scripts.run_evaluation outra_bula` sem editar este arquivo.
    modulo = importlib.import_module(f"app.evaluation.datasets.{nome}")
    return modulo.DATASET


def main() -> None:
    nome_dataset = sys.argv[1] if len(sys.argv) > 1 else "paracetamol_geolab"
    dataset = carregar_dataset(nome_dataset)
    print(f"Dataset: {nome_dataset} ({len(dataset)} casos)\n")

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

    acertos_sem_rerank: list[bool] = []
    acertos_com_rerank: list[bool] = []
    vereditos_geracao: list[str] = []

    for caso in dataset:
        print("=" * 70)
        print(f"PERGUNTA: {caso.pergunta}")
        alvo = caso.pagina_esperada or "N/A (fora de escopo)"

        candidatos = retriever.invoke(caso.pergunta)
        acertou_sem_rerank = hit_rate(candidatos, caso.pagina_esperada)
        acertos_sem_rerank.append(acertou_sem_rerank)

        rerankeados = rerank(reranker, caso.pergunta, candidatos, k_final=K_FINAL)
        acertou_com_rerank = hit_rate(rerankeados, caso.pagina_esperada)
        acertos_com_rerank.append(acertou_com_rerank)

        print(
            f"Retrieval:  cru(k=15)={'OK' if acertou_sem_rerank else 'FALHOU'}  "
            f"|  pós-rerank(k=4)={'OK' if acertou_com_rerank else 'FALHOU'}  "
            f"(página esperada: {alvo})"
        )

        resultado = chain.invoke({"question": caso.pergunta, "chat_history": []})
        resposta = resultado["answer"]
        veredicto = avaliar_resposta_com_llm_juiz(llm, caso.pergunta, caso.resposta_esperada, resposta)
        vereditos_geracao.append(veredicto.veredicto)

        print(f"Resposta:   {resposta}")
        print(f"Veredito:   {veredicto.veredicto} — {veredicto.justificativa}")

    print("\n" + "=" * 70)
    print("RESUMO")
    n = len(dataset)
    print(
        f"Hit Rate cru (k=15, sem reranking):   "
        f"{sum(acertos_sem_rerank) / n:.0%}  ({sum(acertos_sem_rerank)}/{n})"
    )
    print(
        f"Hit Rate pós-reranking (k=4, final):   "
        f"{sum(acertos_com_rerank) / n:.0%}  ({sum(acertos_com_rerank)}/{n})"
    )
    print(f"Vereditos de geração:                 {dict(Counter(vereditos_geracao))}")


if __name__ == "__main__":
    main()
