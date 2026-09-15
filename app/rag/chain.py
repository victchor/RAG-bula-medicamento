from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from sentence_transformers import CrossEncoder

from app.rag.prompts import CONDENSE_QUESTION_PROMPT, RAG_PROMPT
from app.rag.reranker import rerank

K_FINAL = 4


def format_docs(docs: list[Document]) -> str:
    partes = []
    for doc in docs:
        pagina = doc.metadata.get("page_label", "desconhecida")
        partes.append(f"[Página {pagina}]\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


def get_rag_chain(
    retriever: BaseRetriever,
    reranker: CrossEncoder,
    llm: BaseChatModel,
    k_final: int = K_FINAL,
) -> Runnable:
    # Entrada: {"question": str, "chat_history": list[BaseMessage]}
    # Saída: {"answer": str, "source_documents": list[Document]}
    #
    # Os documentos são buscados uma única vez e reaproveitados tanto para
    # montar o contexto quanto para expor as fontes — evita que "fontes
    # exibidas" divirjam dos chunks que de fato geraram a resposta.

    condensar_pergunta = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    def obter_pergunta_para_busca(entrada: dict) -> str:
        # Sem histórico, a pergunta já é autônoma — pula a chamada de
        # reformulação para economizar uma chamada de LLM.
        if not entrada.get("chat_history"):
            return entrada["question"]
        return condensar_pergunta.invoke(entrada)

    def buscar_e_rerankear(entrada: dict) -> list[Document]:
        pergunta_busca = obter_pergunta_para_busca(entrada)
        candidatos = retriever.invoke(pergunta_busca)
        return rerank(reranker, pergunta_busca, candidatos, k_final=k_final)

    buscar_documentos = RunnableParallel(
        question=lambda entrada: entrada["question"],
        chat_history=lambda entrada: entrada.get("chat_history", []),
        source_documents=RunnableLambda(buscar_e_rerankear),
    )

    gerar_resposta = (
        RunnableParallel(
            context=lambda etapa1: format_docs(etapa1["source_documents"]),
            question=lambda etapa1: etapa1["question"],
            chat_history=lambda etapa1: etapa1["chat_history"],
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return buscar_documentos | RunnableParallel(
        answer=gerar_resposta,
        source_documents=lambda etapa1: etapa1["source_documents"],
    )
