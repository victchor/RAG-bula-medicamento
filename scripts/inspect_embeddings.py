# Comparação pergunta-com-pergunta, só para ilustrar similaridade de
# cosseno — o retriever de verdade compara pergunta com chunk de
# documento (scripts/inspect_retriever.py), não pergunta com pergunta.
import sys

import numpy as np

from app.ingestion.embeddings import get_embeddings

sys.stdout.reconfigure(encoding="utf-8")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def main() -> None:
    embeddings = get_embeddings()

    textos = {
        "armazenamento": "Como devo armazenar este medicamento?",
        "temperatura": "Qual a temperatura correta para conservação?",
        "efeitos": "Quais são os efeitos colaterais mais comuns?",
    }

    vetores = {nome: embeddings.embed_query(texto) for nome, texto in textos.items()}

    dimensao = len(next(iter(vetores.values())))
    print(f"Dimensão do vetor: {dimensao}\n")

    pares = [
        ("armazenamento", "temperatura"),
        ("armazenamento", "efeitos"),
        ("temperatura", "efeitos"),
    ]
    for a, b in pares:
        sim = cosine_similarity(vetores[a], vetores[b])
        print(f"similaridade({a}, {b}) = {sim:.4f}")
        print(f"  '{textos[a]}'")
        print(f"  '{textos[b]}'\n")


if __name__ == "__main__":
    main()
