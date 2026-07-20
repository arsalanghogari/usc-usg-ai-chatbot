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
from urllib.parse import urlparse

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PAGES_PATH = BASE_DIR / "pages.json"
KB_PATH = BASE_DIR / "kb.json"
WP_API_BASE = os.getenv("WP_API_BASE", "https://usg.usc.edu/wp-json/wp/v2")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

SCHEMA_SQL = """
create extension if not exists vector;
create table if not exists chunks (
  id bigint generated always as identity primary key,
  source_url text not null,
  source_title text,
  chunk_index int not null,
  text text not null,
  source_modified timestamptz,
  source_modified_year int,
  evergreen boolean not null default false,
  embedding vector(1536),
  unique (source_url, chunk_index)
);
create index if not exists chunks_embedding_hnsw
  on chunks using hnsw (embedding vector_cosine_ops);
"""

UPSERT_SQL = """
insert into chunks (source_url, source_title, chunk_index, text,
                    source_modified, source_modified_year, evergreen, embedding)
values (%s, %s, %s, %s, %s, %s, %s, %s::vector)
on conflict (source_url, chunk_index) do update set
  source_title = excluded.source_title,
  text = excluded.text,
  source_modified = excluded.source_modified,
  source_modified_year = excluded.source_modified_year,
  evergreen = excluded.evergreen,
  embedding = excluded.embedding
"""


def push_to_db(records: List[Dict[str, Any]]) -> None:
    import psycopg

    with psycopg.connect(SUPABASE_DB_URL) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.executemany(
            UPSERT_SQL,
            [
                (
                    r["source_url"], r["source_title"], r["chunk_index"], r["text"],
                    r["source_modified"], r["source_modified_year"], r["evergreen"],
                    json.dumps(r["embedding"]),
                )
                for r in records
            ],
        )
        # Drop rows for pages no longer in the allowlist, and stale tail
        # chunks from pages that shrank since the last crawl.
        urls = sorted({r["source_url"] for r in records})
        cur.execute("delete from chunks where not (source_url = any(%s))", (urls,))
        for url in urls:
            n = sum(1 for r in records if r["source_url"] == url)
            cur.execute(
                "delete from chunks where source_url = %s and chunk_index >= %s",
                (url, n),
            )
    print(f"Upserted {len(records)} chunks into Postgres")

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
                "evergreen": bool(page.get("evergreen")),
            }
        )
    return cleaned


def wp_modified_gmt(url: str) -> str | None:
    """Look up a page/post's last-modified date via the WP REST API by slug.

    Content itself still comes from the live URL (REST content.rendered is
    Divi shortcode soup) — this only fetches the date.
    """
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    if not slug:
        return None
    for endpoint in ("pages", "posts"):
        try:
            r = requests.get(
                f"{WP_API_BASE}/{endpoint}",
                params={"slug": slug, "_fields": "modified_gmt"},
                headers=HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            items = r.json()
            if items:
                return items[0]["modified_gmt"] + "Z"
        except Exception as e:
            print(f"  date lookup ({endpoint}/{slug}) failed: {e}", file=sys.stderr)
    return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = page["url"]
    title = page["title"]

    modified = wp_modified_gmt(url)
    modified_year = int(modified[:4]) if modified else None

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
        for chunk in chunks:
            text = normalize_text(getattr(chunk, "text", ""))
            if not text:
                continue

            metadata = metadata_to_dict(getattr(chunk, "metadata", None))
            records.append(
                {
                    "source_url": url,
                    "source_title": title,
                    # dense numbering (no gaps) — the DB trim of removed tail
                    # chunks relies on it
                    "chunk_index": len(records),
                    "text": text,
                    "source_modified": modified,
                    "source_modified_year": modified_year,
                    "evergreen": page["evergreen"],
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

    if SUPABASE_DB_URL:
        push_to_db(all_records)
    else:
        print("SUPABASE_DB_URL not set, skipping Postgres push", file=sys.stderr)


if __name__ == "__main__":
    main()