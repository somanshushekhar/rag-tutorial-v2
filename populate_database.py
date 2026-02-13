import argparse
import os
import shutil
from pathlib import Path
from pypdf import PdfReader
from typing import List
from dataclasses import dataclass
from get_embedding_function import get_embedding_function
import chromadb

CHROMA_PATH = "chroma"
DATA_PATH = "data"


@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict


def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents() -> List[SimpleDocument]:
    
    docs: List[SimpleDocument] = []
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        print(f"Data directory '{DATA_PATH}' does not exist. Create it and add PDFs.")
        return docs

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            print(f"Failed to read {pdf_path}: {e}")
            continue

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            metadata = {"source": str(pdf_path), "page": page_num}
            docs.append(SimpleDocument(page_content=text, metadata=metadata))

    return docs


def split_documents(documents: List[SimpleDocument], chunk_size: int = 800, chunk_overlap: int = 80) -> List[SimpleDocument]:
    chunks: List[SimpleDocument] = []

    for doc in documents:
        text = doc.page_content or ""
        if not text:
            continue
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunk_text = text[start:end]
            chunk_meta = dict(doc.metadata)
            chunks.append(SimpleDocument(page_content=chunk_text, metadata=chunk_meta))
            start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def add_to_chroma(chunks: List[SimpleDocument]):
    # Initialize chromadb client with persistence.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="documents")

    # Calculate IDs for chunks.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Fetch existing IDs from collection.
    try:
        existing = collection.get(include=["ids"])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Prepare new items.
    new_texts = []
    new_metadatas = []
    new_ids = []
    skipped_count = 0

    for chunk in chunks_with_ids:
        cid = chunk.metadata.get("id")
        if cid in existing_ids:
            skipped_count += 1
            continue
        new_texts.append(chunk.page_content)
        new_metadatas.append(chunk.metadata)
        new_ids.append(cid)

    print(f"Skipped {skipped_count} existing chunks")
    print(f"Adding {len(new_texts)} new chunks")

    if not new_texts:
        print("✅ No new documents to add")
        return

    # Compute embeddings using the embedding function provided.
    embedding_fn = get_embedding_function()
    try:
        embeddings = embedding_fn.embed_documents(new_texts)
    except AttributeError:
        # Some embedding wrappers use 'embed' or are callable.
        try:
            embeddings = [embedding_fn(text) for text in new_texts]
        except Exception as e:
            raise RuntimeError(f"Failed to compute embeddings: {e}")

    # Add to collection in batches to avoid large memory spikes.
    batch_size = 256
    for i in range(0, len(new_texts), batch_size):
        batch_texts = new_texts[i : i + batch_size]
        batch_metadatas = new_metadatas[i : i + batch_size]
        batch_ids = new_ids[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        collection.add(documents=batch_texts, metadatas=batch_metadatas, ids=batch_ids, embeddings=batch_embeddings)

    # ChromaDB 0.4+ automatically persists with PersistentClient
    print(f"👉 Added new documents: {len(new_texts)}")


def calculate_chunk_ids(chunks: List[SimpleDocument]) -> List[SimpleDocument]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()