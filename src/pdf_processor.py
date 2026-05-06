"""
Extract text and images from all PDF files in documents/.
Saves one JSON file per document to outputs/dataset/raw_text/.
Images are processed via Claude (if API key is configured) or skipped.

Usage:
    python src/pdf_processor.py
"""
import sys
import json
from pathlib import Path

# Allow sibling imports when run directly
sys.path.insert(0, str(Path(__file__).parent))

import fitz  # pymupdf
from image_processor import describe_image, is_available as image_api_available

ROOT = Path(__file__).parent.parent
DOCUMENTS_DIR = ROOT / "documents"
RAW_TEXT_DIR = ROOT / "outputs" / "dataset" / "raw_text"
RAW_IMAGES_DIR = ROOT / "outputs" / "dataset" / "raw_images"


def _extract_pdf(pdf_path: Path) -> dict:
    """Return a structured dict with text and image descriptions for every page."""
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()
        image_descriptions = []

        for img_idx, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Persist the raw image for reference
            img_filename = f"{pdf_path.stem}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
            (RAW_IMAGES_DIR / img_filename).write_bytes(image_bytes)

            description = describe_image(image_bytes, image_ext)
            if description:
                image_descriptions.append(description)

        pages.append(
            {
                "page": page_num + 1,
                "text": page_text,
                "image_descriptions": image_descriptions,
            }
        )

    doc.close()
    return {"source": pdf_path.name, "pages": pages}


def process_all_documents() -> list[dict]:
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in documents/. Add your PDFs there and re-run.")
        return []

    if not image_api_available():
        print(
            "Note: ANTHROPIC_API_KEY is not set — images inside PDFs will be skipped.\n"
            "      Set ANTHROPIC_API_KEY in .env to enable image processing via Claude."
        )

    print(f"Found {len(pdf_files)} PDF file(s).\n")
    all_data = []

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        data = _extract_pdf(pdf_path)
        all_data.append(data)

        output_path = RAW_TEXT_DIR / f"{pdf_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        n_pages = len(data["pages"])
        n_images = sum(len(p["image_descriptions"]) for p in data["pages"])
        n_chars = sum(len(p["text"]) for p in data["pages"])
        print(f"    -> {n_pages} pages | {n_chars:,} chars of text | {n_images} images described")

    print(f"\nDone. Raw data saved to {RAW_TEXT_DIR}")
    return all_data


if __name__ == "__main__":
    process_all_documents()
