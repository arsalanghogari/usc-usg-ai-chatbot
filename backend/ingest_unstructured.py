#!/usr/bin/env python3
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from openai import OpenAI
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PAGES_PATH = BASE_DIR / "pages.json"
KB_PATH = BASE_DIR / "kb.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    print("Set OPENAI_API_KEY in your .env file or environment.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def metadata_to_dict(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if hasattr(metadata, "model_dump"):
        try:
            return metadata.model_dump()
        except Exception:
            pass
    if hasattr(metadata, "to_dict"):
        try:
            return metadata.to_dict()
        except Exception:
            pass
    if hasattr(metadata, "__dict__"):
        return {k: v for k, v in vars(metadata).items() if not k.startswith("_")}
    return {"value": str(metadata)}


def load_pages() -> List[Dict[str, str]]:
    if not PAGES_PATH.exists():
        raise FileNotFoundError(f"Missing {PAGES_PATH}")

    pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
    if not isinstance(pages, list):
        raise ValueError("pages.json must contain a list")

    cleaned: List[Dict[str, str]] = []
    for i, page in enumerate(pages):
        if not isinstance(page, dict) or not page.get("url"):
            raise ValueError(f"Invalid page at index {i}: {page}")
        cleaned.append(
            {
                "url": page["url"],
                "title": page.get("title") or page["url"],
            }
        )
    return cleaned


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_page(page: Dict[str, str]) -> List[Dict[str, Any]]:
    url = page["url"]
    title = page["title"]

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(response.text)
            tmp_path = tmp.name

        elements = partition_html(filename=tmp_path)
        chunks = chunk_by_title(elements, max_characters=1200)

        records: List[Dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            text = normalize_text(getattr(chunk, "text", ""))
            if not text:
                continue

            metadata = metadata_to_dict(getattr(chunk, "metadata", None))
            records.append(
                {
                    "source_url": url,
                    "source_title": title,
                    "chunk_index": chunk_index,
                    "text": text,
                    "metadata": metadata,
                }
            )

        return records

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    pages = load_pages()
    all_records: List[Dict[str, Any]] = []

    for page in pages:
        try:
            page_records = ingest_page(page)
            all_records.extend(page_records)
            print(f"Ingested {len(page_records)} chunks from {page['url']}")
        except Exception as e:
            print(f"Skipping {page['url']}: {e}", file=sys.stderr)

    if not all_records:
        raise RuntimeError("No chunks were produced. Check your URLs and page access.")

    embeddings = embed_texts([record["text"] for record in all_records])
    for record, embedding in zip(all_records, embeddings):
        record["embedding"] = embedding

    kb = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "chunks": all_records,
    }

    KB_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {KB_PATH} with {len(all_records)} chunks")


if __name__ == "__main__":
    main()