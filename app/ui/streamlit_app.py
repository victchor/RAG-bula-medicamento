import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.config import UPLOADS_DIR, VECTORSTORE_DIR, load_settings
from app.ingestion.embeddings import get_embeddings
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.vectorstore import create_new_vectorstore
from app.llm import get_chat_model
from app.observability.handler import ObservabilityHandler, formatar_metricas
from app.rag.chain import get_rag_chain
from app.rag.reranker import get_reranker
from app.rag.retriever import get_retriever
from app.security.file_validation import sanitizar_nome_arquivo, validar_pdf

st.set_page_config(page_title="Chatbot de Bulas", page_icon="💊")

MAX_HISTORICO_MENSAGENS = 6
MAX_TAMANHO_PERGUNTA = 500


# Streamlit reexecuta o script inteiro a cada interação do usuário.
# st.cache_resource memoriza o resultado entre essas execuções, para não
# recarregar modelos pesados a cada clique.
@st.cache_resource
def carregar_llm():
    settings = load_settings()
    return get_chat_model(settings)


@st.cache_resource
def carregar_embeddings():
    return get_embeddings()


@st.cache_resource
def carregar_reranker():
    return get_reranker()


def montar_chat_history() -> list[HumanMessage | AIMessage]:
    mensagens_recentes = st.session_state.messages[-MAX_HISTORICO_MENSAGENS:]
    historico: list[HumanMessage | AIMessage] = []
    for msg in mensagens_recentes:
        if msg["role"] == "user":
            historico.append(HumanMessage(content=msg["content"]))
        else:
            historico.append(AIMessage(content=msg["content"]))
    return historico


def processar_bula(arquivo_pdf) -> bool:
    conteudo = arquivo_pdf.getvalue()
    validacao = validar_pdf(arquivo_pdf.name, conteudo)
    if not validacao.valido:
        st.error(f"Arquivo rejeitado: {validacao.erro}")
        return False

    nome_seguro = sanitizar_nome_arquivo(arquivo_pdf.name)
    caminho = UPLOADS_DIR / nome_seguro
    caminho.write_bytes(conteudo)

    try:
        with st.spinner("Lendo PDF, dividindo em chunks e gerando embeddings..."):
            documentos = load_pdf(caminho)
            if not documentos or not any(d.page_content.strip() for d in documentos):
                st.error(
                    "Não foi possível extrair texto deste PDF — pode ser um PDF "
                    "escaneado (imagem) sem OCR, ou um arquivo corrompido."
                )
                return False

            chunks = split_documents(documentos)
            embeddings = carregar_embeddings()
            vectorstore = create_new_vectorstore(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(VECTORSTORE_DIR),
            )
    except Exception:
        st.error("Não foi possível processar este PDF. Tente outro arquivo.")
        return False

    retriever = get_retriever(vectorstore)
    reranker = carregar_reranker()
    chain = get_rag_chain(retriever, reranker, carregar_llm())

    st.session_state.chain = chain
    st.session_state.bula_nome = nome_seguro
    st.session_state.messages = []
    return True


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.bula_nome = None


st.title("💊 Chatbot de Bulas de Medicamentos")
st.caption(
    "Assistente de **consulta** ao texto de bulas — não realiza diagnóstico médico "
    "nem substitui orientação profissional. Em caso de dúvida, consulte um médico "
    "ou farmacêutico."
)

with st.sidebar:
    st.header("1. Envie a bula")
    arquivo = st.file_uploader("PDF da bula", type="pdf")
    if arquivo is not None and st.button("Processar bula"):
        if processar_bula(arquivo):
            st.success(f"Bula processada: {st.session_state.bula_nome}")

    if st.session_state.bula_nome:
        st.info(f"Bula ativa: {st.session_state.bula_nome}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fontes"):
            with st.expander("Fontes utilizadas"):
                for fonte in msg["fontes"]:
                    st.markdown(f"**Página {fonte['pagina']}**")
                    st.text(fonte["trecho"])
        if msg.get("metricas"):
            st.caption(msg["metricas"])

pergunta = st.chat_input("Faça uma pergunta sobre a bula...")

if pergunta:
    if st.session_state.chain is None:
        st.warning("Envie e processe uma bula antes de perguntar.")
    elif len(pergunta) > MAX_TAMANHO_PERGUNTA:
        st.warning(f"Pergunta muito longa (máximo {MAX_TAMANHO_PERGUNTA} caracteres).")
    else:
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando na bula e gerando resposta..."):
                    # Histórico construído antes de adicionar a pergunta
                    # atual: representa a conversa até o turno anterior.
                    chat_history = montar_chat_history()

                    observador = ObservabilityHandler()
                    inicio = time.perf_counter()
                    resultado = st.session_state.chain.invoke(
                        {"question": pergunta, "chat_history": chat_history},
                        config={"callbacks": [observador]},
                    )
                    duracao = time.perf_counter() - inicio
                    resposta = resultado["answer"]
            except Exception:
                st.error(
                    "Não foi possível gerar uma resposta agora (erro de conexão com a "
                    "LLM ou limite de uso atingido). Tente novamente em instantes."
                )
            else:
                st.markdown(resposta)
                fontes = [
                    {
                        "pagina": doc.metadata.get("page_label", "?"),
                        "trecho": doc.page_content[:300],
                    }
                    for doc in resultado["source_documents"]
                ]
                with st.expander("Fontes utilizadas"):
                    for fonte in fontes:
                        st.markdown(f"**Página {fonte['pagina']}**")
                        st.text(fonte["trecho"])

                settings = load_settings()
                nome_modelo = (
                    settings.claude_model if settings.llm_provider == "anthropic" else settings.gemini_model
                )
                paginas_finais = [f["pagina"] for f in fontes]
                metricas_texto = formatar_metricas(observador.metricas, duracao, nome_modelo, paginas_finais)
                st.caption(metricas_texto)

                st.session_state.messages.append(
                    {"role": "assistant", "content": resposta, "fontes": fontes, "metricas": metricas_texto}
                )
