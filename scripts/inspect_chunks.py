import sys

from app.config import UPLOADS_DIR
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.splitter import split_documents

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.inspect_chunks <nome_do_arquivo.pdf>")
        sys.exit(1)

    pdf_path = UPLOADS_DIR / sys.argv[1]
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    print(f"Páginas originais: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")
    tamanhos = [len(c.page_content) for c in chunks]
    print(f"Tamanho médio: {sum(tamanhos) / len(tamanhos):.0f} caracteres")
    print(f"Tamanho min/max: {min(tamanhos)} / {max(tamanhos)}\n")

    for i, chunk in enumerate(chunks[:5]):
        print("=" * 60)
        print(f"Chunk {i} | página {chunk.metadata.get('page_label')} | {len(chunk.page_content)} caracteres")
        print("-" * 60)
        print(chunk.page_content)
        print()


if __name__ == "__main__":
    main()
