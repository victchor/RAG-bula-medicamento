import sys

from app.config import UPLOADS_DIR
from app.ingestion.pdf_loader import load_pdf

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.inspect_pdf <nome_do_arquivo.pdf>")
        print(f"O arquivo deve estar em: {UPLOADS_DIR}")
        sys.exit(1)

    pdf_path = UPLOADS_DIR / sys.argv[1]
    if not pdf_path.exists():
        print(f"Arquivo não encontrado: {pdf_path}")
        sys.exit(1)

    documents = load_pdf(pdf_path)

    print(f"Total de páginas extraídas: {len(documents)}\n")
    for doc in documents[:3]:
        print("=" * 60)
        print(f"Metadata: {doc.metadata}")
        print(f"Tamanho do texto: {len(doc.page_content)} caracteres")
        print("-" * 60)
        print(doc.page_content[:500])
        print()


if __name__ == "__main__":
    main()
