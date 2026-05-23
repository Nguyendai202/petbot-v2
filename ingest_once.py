"""Run once to index the PDF into Qdrant. Safe to re-run."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from vetbot import ingest_pdf

PDF = Path(__file__).parent / "data" / "dog-and-cat disease.pdf"

if __name__ == "__main__":
    print(f"Ingesting: {PDF}")
    stats = ingest_pdf(str(PDF))
    print(f"Done → chunks: {stats['chunks_created']}, files: {stats['processed']}")