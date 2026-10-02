from pathlib import Path
import re

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_text(text):
    text = text.replace("\u00ad", "")

    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path, output_path):
    reader = PdfReader(str(pdf_path))

    cleaned_pages = []
    total_pages = len(reader.pages)

    print(f"\nReading: {pdf_path.name}")
    print(f"Total pages: {total_pages}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            cleaned_pages.append(clean_text(text))

        if page_number % 25 == 0 or page_number == total_pages:
            print(f"Processed {page_number}/{total_pages} pages")

    complete_text = "\n\n".join(cleaned_pages)
    output_path.write_text(complete_text, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1_000_000

    print(f"Saved: {output_path.name}")
    print(f"Text size: {size_mb:.2f} MB")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in {RAW_DIR}"
        )

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        output_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
        extract_pdf(pdf_path, output_path)

    print("\nCorpus preparation finished.")


if __name__ == "__main__":
    main()