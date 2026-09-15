from dataclasses import dataclass


@dataclass
class CasoDeTeste:
    pergunta: str
    resposta_esperada: str
    pagina_esperada: str | None  # None = pergunta fora do escopo da bula
