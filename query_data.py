import argparse
import os
import json
import shutil
import subprocess
from get_embedding_function import get_embedding_function
import chromadb
from chromadb.config import Settings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def generate_with_ollama(prompt: str, model: str, base_url: str) -> str:
    # Try CLI first if available
    if shutil.which("ollama"):
        try:
            # Pass the prompt as an argument to the CLI
            res = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                return res.stdout.strip()
            # otherwise fall back to HTTP
        except Exception:
            pass

    # Fallback to HTTP generate endpoint
    import requests

    url = f"{base_url.rstrip('/')}/api/generate"
    try:
        resp = requests.post(url, json={"model": model, "prompt": prompt}, stream=True, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to call Ollama generate endpoint: {e}")

    # Streamed NDJSON response; collect text deltas when possible
    output = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            output += line
            continue

        # Common shapes: {"type":"response.delta","delta":"text"} or nested
        delta = None
        if isinstance(obj, dict):
            delta = obj.get("delta") or obj.get("text") or obj.get("content")
            if isinstance(delta, dict):
                # try nested content
                delta = delta.get("content") or delta.get("text")
        if delta:
            output += delta

    return output


def query_rag(query_text: str):
    # Prepare the DB and embedding function.
    embedding_function = get_embedding_function()

    # Initialize chroma client and collection
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH)
    client = chromadb.Client(settings=settings)
    collection = client.get_collection(name="documents")

    # Compute query embedding
    try:
        query_embedding = embedding_function.embed_documents([query_text])[0]
    except AttributeError:
        query_embedding = embedding_function(text=query_text)

    # Query chroma
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["metadatas", "documents", "distances"])

    # results contains lists per query; take first
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Extract ids from metadatas if present
    ids = []
    for m in metadatas:
        if isinstance(m, dict) and "id" in m:
            ids.append(m.get("id"))
        else:
            # fallback to source/page if available
            src = None
            if isinstance(m, dict):
                src = m.get("source")
                page = m.get("page")
                if src and page is not None:
                    ids.append(f"{src}:{page}")
                elif src:
                    ids.append(src)
                else:
                    ids.append(None)
            else:
                ids.append(None)

    context_text = "\n\n---\n\n".join(docs)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    ollama_model = os.environ.get("OLLAMA_MODEL", "mistral")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    response_text = generate_with_ollama(prompt, ollama_model, ollama_base_url)
    response_text_clean = response_text.strip() if isinstance(response_text, str) else ""

    # Build human-readable sources list
    readable_sources = []
    for i, doc in enumerate(docs):
        src = ids[i] if i < len(ids) else None
        dist = distances[i] if i < len(distances) else None
        snippet = doc.strip().replace("\n", " ")[:300]
        readable_sources.append({
            "rank": i + 1,
            "source": src,
            "distance": round(float(dist), 6) if dist is not None else None,
            "snippet": snippet,
        })

    # Print human readable output
    print("\n=== Answer ===\n")
    if response_text_clean:
        print(response_text_clean)
    else:
        print("(no response returned by the LLM)")

    print("\n=== Top Sources ===\n")
    for item in readable_sources:
        print(f"{item['rank']}. {item['source']} (distance: {item['distance']})")
        print(f"   Snippet: {item['snippet']}\n")

    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
