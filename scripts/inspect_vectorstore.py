import shutil
import sys

from app.config import UPLOADS_DIR, VECTORSTORE_DIR
from app.ingestion.embeddings import get_embeddings
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.vectorstore import create_new_vectorstore, load_existing_vectorstore

sys.stdout.reconfigure(encoding="utf-8")


def limpar_vectorstore_antigo() -> None:
    for item in VECTORSTORE_DIR.iterdir():
        if item.name == ".gitkeep":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def main() -> None:
    pdf_path = UPLOADS_DIR / "PARACETAMOL-Bula-Profissional.pdf"
    chunks = split_documents(load_pdf(pdf_path))
    embeddings = get_embeddings()

    limpar_vectorstore_antigo()

    create_new_vectorstore(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"{len(chunks)} chunks indexados e persistidos em {VECTORSTORE_DIR}\n")

    vectorstore = load_existing_vectorstore(
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    pergunta = "Como devo armazenar este medicamento?"
    resultados = vectorstore.similarity_search(pergunta, k=4)

    print(f"Busca (k=4): '{pergunta}'\n")
    for i, doc in enumerate(resultados, start=1):
        print(f"{i}. página {doc.metadata.get('page_label')} | {doc.page_content[:150]!r}")


if __name__ == "__main__":
    main()
