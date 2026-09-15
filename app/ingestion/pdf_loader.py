from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: Path) -> list[Document]:
    # Um Document por página, cada um com a página de origem em metadata
    # (usado depois para citar a fonte nas respostas).
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()
