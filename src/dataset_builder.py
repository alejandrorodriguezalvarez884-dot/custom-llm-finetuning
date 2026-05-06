"""
Build the SFT training dataset from processed PDF content.

Two modes:
  chunks (default, no API key needed)
    Splits document text into chunks and wraps each one in a ChatML
    instruction envelope. Fast and offline.

  qa (requires ANTHROPIC_API_KEY)
    Uses Claude to generate question-answer pairs from each chunk.
    Produces a dataset that is much better suited for a chat use case.
    Falls back to chunks mode automatically if the key is missing.

Usage:
    python src/dataset_builder.py              # auto-selects mode
    python src/dataset_builder.py --mode qa    # force Q&A mode
    python src/dataset_builder.py --mode chunks
    python src/dataset_builder.py --mode qa --n-questions 8
"""
import sys
import json
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
RAW_TEXT_DIR = ROOT / "outputs" / "dataset" / "raw_text"
DATASET_PATH = ROOT / "outputs" / "dataset" / "train.jsonl"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_SEQ_LEN = int(os.getenv("FINETUNE_MAX_SEQ_LEN", "512"))

# Approximate words per chunk (1 word ≈ 1.3 tokens)
_CHUNK_WORDS = int(MAX_SEQ_LEN * 0.6)
_CHUNK_OVERLAP = 40

_SYSTEM_PROMPT = (
    "You are a helpful assistant with specialized knowledge from a curated set of documents. "
    "Answer questions accurately and concisely based on your training. "
    "If you are unsure, say so clearly."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + _CHUNK_WORDS])
        if len(chunk.strip()) >= 80:
            chunks.append(chunk.strip())
        i += _CHUNK_WORDS - _CHUNK_OVERLAP
    return chunks


def _chatml(user_msg: str, assistant_msg: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


def _full_page_text(page: dict) -> str:
    parts = [page["text"]]
    for desc in page.get("image_descriptions", []):
        parts.append(f"[Image content: {desc}]")
    return "\n\n".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Chunks mode
# ---------------------------------------------------------------------------

def _build_chunks(documents: list[dict]) -> list[dict]:
    examples = []
    for doc in documents:
        for page in doc["pages"]:
            full_text = _full_page_text(page)
            for chunk in _chunk_text(full_text):
                text = _chatml(
                    "Based on your knowledge from the documents, explain the following topic in detail.",
                    chunk,
                )
                examples.append({"text": text})
    return examples


# ---------------------------------------------------------------------------
# Q&A mode
# ---------------------------------------------------------------------------

def _build_qa(documents: list[dict], n_questions: int) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    examples = []

    for doc in documents:
        print(f"  Generating Q&A for: {doc['source']}")
        for page in doc["pages"]:
            full_text = _full_page_text(page)
            for chunk in _chunk_text(full_text):
                try:
                    response = client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=2048,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Based on the text below, generate {n_questions} diverse "
                                    f"question-answer pairs. Questions must be specific and "
                                    f"answers must be detailed and accurate. "
                                    f"Output ONLY a JSON array with objects containing "
                                    f"'question' and 'answer' keys. No extra text.\n\n"
                                    f"Text:\n{chunk}"
                                ),
                            }
                        ],
                    )
                    raw = response.content[0].text.strip()
                    start, end = raw.find("["), raw.rfind("]") + 1
                    if start == -1 or end == 0:
                        continue
                    pairs = json.loads(raw[start:end])
                    for pair in pairs:
                        q, a = pair.get("question", ""), pair.get("answer", "")
                        if q and a:
                            examples.append({"text": _chatml(q, a)})
                except Exception as exc:
                    print(f"    Warning: Q&A generation failed for a chunk — {exc}")

    return examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the SFT training dataset.")
    parser.add_argument(
        "--mode",
        choices=["qa", "chunks"],
        default=None,
        help="Dataset generation mode (default: qa if API key present, else chunks).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=5,
        dest="n_questions",
        help="Number of Q&A pairs to generate per text chunk (qa mode only).",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        mode = "qa" if ANTHROPIC_API_KEY else "chunks"

    if mode == "qa" and not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY is not set. Falling back to chunks mode.")
        mode = "chunks"

    json_files = sorted(RAW_TEXT_DIR.glob("*.json"))
    if not json_files:
        print("No processed documents found. Run pdf_processor.py first.")
        sys.exit(1)

    documents = []
    for f in json_files:
        with open(f, encoding="utf-8") as fp:
            documents.append(json.load(fp))

    print(f"Building dataset in '{mode}' mode from {len(documents)} document(s)...")

    if mode == "qa":
        examples = _build_qa(documents, args.n_questions)
    else:
        examples = _build_chunks(documents)

    if not examples:
        print("No training examples generated. Check that your documents contain text.")
        sys.exit(1)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDataset saved: {DATASET_PATH}")
    print(f"Total training examples: {len(examples)}")


if __name__ == "__main__":
    main()
