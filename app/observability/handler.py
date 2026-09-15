# Callback do LangChain: qualquer Runnable (retriever, LLM, chain) aceita
# `config={"callbacks": [...]}` no .invoke() e notifica cada handler
# registrado a cada evento interno (chamada de LLM, resultado do
# retriever, erros) — sem exigir nenhuma mudança em app/rag/chain.py.
from dataclasses import dataclass, field
from typing import Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# Preço por 1 milhão de tokens (USD). Não inclui prompt caching, desconto de
# batelada, nem variações regionais.
PRECOS_POR_MILHAO_TOKENS = {
    "claude-haiku-4-5-20251001": {"entrada": 1.00, "saida": 5.00},
    "claude-sonnet-5": {"entrada": 2.00, "saida": 10.00},
    "claude-opus-5": {"entrada": 5.00, "saida": 25.00},
    "gemini-3.1-flash-lite": {"entrada": 0.00, "saida": 0.00},  # camada gratuita
    "gemini-3.1-flash-lite-paga": {"entrada": 0.25, "saida": 1.50},
}


@dataclass
class Metricas:
    chamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    paginas_candidatas: list[str] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)

    def custo_estimado_usd(self, modelo: str) -> float:
        precos = PRECOS_POR_MILHAO_TOKENS.get(modelo)
        if precos is None:
            return 0.0
        custo_entrada = (self.tokens_entrada / 1_000_000) * precos["entrada"]
        custo_saida = (self.tokens_saida / 1_000_000) * precos["saida"]
        return custo_entrada + custo_saida


class ObservabilityHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metricas = Metricas()

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID, **kwargs
    ) -> None:
        self.metricas.chamadas_llm += 1

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs) -> None:
        for geracoes in response.generations:
            for geracao in geracoes:
                mensagem = getattr(geracao, "message", None)
                uso = getattr(mensagem, "usage_metadata", None) if mensagem else None
                if uso:
                    self.metricas.tokens_entrada += uso.get("input_tokens", 0)
                    self.metricas.tokens_saida += uso.get("output_tokens", 0)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        self.metricas.erros.append(f"LLM: {error}")

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs) -> None:
        # O reranking não é um Runnable rastreado, então este evento reflete
        # os candidatos da busca vetorial, não os finais pós-reranking (ver
        # `source_documents` no retorno da chain para os finais).
        self.metricas.paginas_candidatas = [doc.metadata.get("page_label") for doc in documents]

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        self.metricas.erros.append(f"Retriever: {error}")

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        self.metricas.erros.append(f"Chain: {error}")


def formatar_metricas(
    metricas: Metricas, duracao_segundos: float, modelo: str, paginas_finais: list[str] | None = None
) -> str:
    custo = metricas.custo_estimado_usd(modelo)
    linha = (
        f"tempo={duracao_segundos:.2f}s | "
        f"chamadas_llm={metricas.chamadas_llm} | "
        f"tokens(entrada/saída)={metricas.tokens_entrada}/{metricas.tokens_saida} | "
        f"candidatos(k=15)={metricas.paginas_candidatas} | "
        f"custo_estimado=${custo:.5f}"
    )
    if paginas_finais is not None:
        linha += f" | usados_na_resposta={paginas_finais}"
    if metricas.erros:
        linha += f" | ERROS={metricas.erros}"
    return linha
