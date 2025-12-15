import os
import requests
from typing import List


class OllamaEmbeddingsWrapper:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call the local Ollama embed API to get embeddings for a list of texts."""
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Expected shape: {"embeddings": [[...], ...]}
        embeddings = data.get("embeddings")
        if embeddings is None:
            # Some Ollama versions may return the list directly
            return data
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedding_function():
    # Use Ollama embeddings running on your local Ollama server (no API keys required).
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return OllamaEmbeddingsWrapper(model=model, base_url=base_url)
