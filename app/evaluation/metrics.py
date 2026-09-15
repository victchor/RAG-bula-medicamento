from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def hit_rate(documentos_recuperados: list[Document], pagina_esperada: str | None) -> bool:
    if pagina_esperada is None:
        return True

    paginas_recuperadas = {doc.metadata.get("page_label") for doc in documentos_recuperados}
    return pagina_esperada in paginas_recuperadas


JUDGE_SYSTEM_PROMPT = """Você é um avaliador de qualidade de um sistema de perguntas e respostas \
sobre bulas de medicamentos.

Você vai receber a pergunta, uma resposta esperada (gabarito, escrita por um humano) e a resposta \
gerada pelo sistema sendo avaliado. Compare as duas quanto ao CONTEÚDO — não exija texto idêntico, \
frases diferentes com o mesmo significado contam como corretas.

Responda em EXATAMENTE este formato, com duas linhas:
VEREDITO: CORRETO ou PARCIAL ou INCORRETO
JUSTIFICATIVA: uma frase curta explicando o veredito

Critérios:
- CORRETO: a resposta cobre a informação principal do gabarito, sem contradizê-lo.
- PARCIAL: cobre só parte da informação, ou está correta mas incompleta.
- INCORRETO: contradiz o gabarito, ou — quando o gabarito diz que a informação não existe na \
bula — o sistema inventou uma resposta em vez de dizer que não encontrou."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", "Pergunta: {pergunta}\n\nGabarito: {gabarito}\n\nResposta do sistema: {resposta}"),
    ]
)


@dataclass
class VeredictoJuiz:
    veredicto: str
    justificativa: str


def avaliar_resposta_com_llm_juiz(
    llm: BaseChatModel, pergunta: str, gabarito: str, resposta: str
) -> VeredictoJuiz:
    chain_juiz = JUDGE_PROMPT | llm | StrOutputParser()
    saida = chain_juiz.invoke({"pergunta": pergunta, "gabarito": gabarito, "resposta": resposta})

    # Parsing tolerante: LLMs nem sempre seguem instrução de formato à risca.
    veredicto, justificativa = "DESCONHECIDO", saida.strip()
    for linha in saida.strip().splitlines():
        if linha.upper().startswith("VEREDITO"):
            veredicto = linha.split(":", 1)[1].strip().upper()
        elif linha.upper().startswith("JUSTIFICATIVA"):
            justificativa = linha.split(":", 1)[1].strip()

    return VeredictoJuiz(veredicto=veredicto, justificativa=justificativa)
