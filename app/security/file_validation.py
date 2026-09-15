import os
import re
from dataclasses import dataclass

MAX_UPLOAD_SIZE_MB = 20

# Assinatura real de um PDF (primeiros bytes do arquivo). Checar só a
# extensão do nome seria fácil de forjar renomeando qualquer arquivo.
PDF_MAGIC_BYTES = b"%PDF-"


@dataclass
class ResultadoValidacao:
    valido: bool
    erro: str | None = None


def validar_pdf(nome_arquivo: str, conteudo: bytes) -> ResultadoValidacao:
    tamanho_mb = len(conteudo) / (1024 * 1024)
    if tamanho_mb > MAX_UPLOAD_SIZE_MB:
        return ResultadoValidacao(
            valido=False,
            erro=f"Arquivo muito grande ({tamanho_mb:.1f}MB). Limite: {MAX_UPLOAD_SIZE_MB}MB.",
        )

    if not nome_arquivo.lower().endswith(".pdf"):
        return ResultadoValidacao(valido=False, erro="Extensão de arquivo inválida — só .pdf é aceito.")

    if not conteudo.startswith(PDF_MAGIC_BYTES):
        return ResultadoValidacao(
            valido=False,
            erro="O conteúdo do arquivo não parece ser um PDF de verdade (assinatura de arquivo incorreta).",
        )

    return ResultadoValidacao(valido=True)


def sanitizar_nome_arquivo(nome_arquivo: str) -> str:
    # os.path.basename remove qualquer componente de caminho — protege
    # contra um nome como "../../../.env" escrever fora de data/uploads/.
    nome_base = os.path.basename(nome_arquivo)
    nome_seguro = re.sub(r"[^A-Za-z0-9._-]", "_", nome_base)
    return nome_seguro or "bula.pdf"
