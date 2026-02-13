# ?? Aflac Assist Chat

A Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, ChromaDB, and Ollama for intelligent document Q&A.

## ?? Features

- **?? PDF Upload & Ingestion**: Upload multiple PDFs and automatically index them
- **?? Real-time Chat Interface**: Modern, responsive chat UI with streaming responses
- **?? Semantic Search**: Uses vector embeddings for accurate document retrieval
- **?? AI-Powered Answers**: Leverages local LLMs via Ollama
- **?? Source Citations**: Shows which documents were used to generate answers
- **?? Incremental Updates**: Add new documents without resetting the entire database
- **?? Beautiful UI**: Gradient design with smooth animations

## ?? Quick Start

### Prerequisites

- Python 3.11 (recommended) or 3.12
- [Ollama](https://ollama.ai/) installed and running
- Git

## Installation

1. Create a virtual environment (recommended):

   ```bash
   # Windows PowerShell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Pull required Ollama models:

   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral
   # Or: ollama pull llama3.2
   ```

4. Run the application:

   ```bash
   python -m uvicorn app:app --reload --port 8000
   ```

5. Open your browser:
   - Upload documents: http://localhost:8000
   - Chat interface: http://localhost:8000/chat

## ?? Usage

### Upload Documents

1. Navigate to http://localhost:8000
2. Select one or more PDF files
3. (Optional) Check "Reset database" to clear existing data
4. Click "Upload & Ingest"
5. Monitor progress in the terminal

### Chat with Your Documents

1. Go to http://localhost:8000/chat
2. Type your question
3. Get AI-powered answers with source citations
4. Responses stream in real-time!

### CLI Query (Optional)

Run a query from the command line:

```bash
python query_data.py "What is the main topic?"
```

## ?? Project Structure

- `app.py` - FastAPI application with chat and upload endpoints
- `data/` - PDF files storage directory
- `chroma/` - ChromaDB vector database
- `get_embedding_function.py` - Ollama embedding wrapper
- `populate_database.py` - Document ingestion and chunking
- `query_data.py` - RAG query logic with streaming support
- `templates/` - HTML templates for UI
  - `index.html` - Upload page
  - `chat.html` - Real-time chat interface
  - `result.html` - Query result page
- `test_rag.py` - Test harness
- `requirements.txt` - Python dependencies

## ?? Configuration

Set environment variables to customize:

```bash
# Embedding model (default: nomic-embed-text)
export OLLAMA_EMBED_MODEL="nomic-embed-text"

# LLM model (default: mistral)
export OLLAMA_MODEL="llama3.2"

# Ollama base URL (default: http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"
```

## ?? Testing

Run the setup test:

```bash
python test_setup.py
```

## ??? Technology Stack

- **Backend**: FastAPI
- **Vector Database**: ChromaDB
- **LLM**: Ollama (Mistral, Llama3.2, etc.)
- **Embeddings**: Nomic Embed Text
- **PDF Processing**: pypdf
- **Frontend**: HTML/CSS/JavaScript (Vanilla)

## ?? Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ?? License

This project is open source and available under the MIT License.

## ????? Author

**Somansh Shekhar**
- GitHub: [@somanshushekhar](https://github.com/somanshushekhar)

## ?? Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [Ollama](https://ollama.ai/)
- Vector search by [ChromaDB](https://www.trychroma.com/)

---

? Star this repo if you find it helpful!
This is a small tutorial repository — feel free to open issues or submit PRs to improve examples or add additional providers.

## License

This repository does not include an explicit license file. Add one if you plan to reuse or publish this project.
