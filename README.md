# rag-tutorial-v2

A minimal Retrieval-Augmented Generation (RAG) tutorial project using LangChain and ChromaDB. The repository shows how to:

- Extract text from PDFs and chunk it for vector indexing (`populate_database.py`).
- Index embeddings with ChromaDB and query the index (`query_data.py`).
- Run simple automated checks against expected answers (`test_rag.py`).

## Prerequisites

- Python 3.9+
- pip
- Either Ollama or AWS credentials for Bedrock depending on the embedding/LLM provider you choose.

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   . .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project layout

- `data/` - Place PDF files here to be indexed.
- `chroma/` - ChromaDB persistence directory created by `populate_database.py`.
- `get_embedding_function.py` - Select and configure the embedding implementation (Bedrock, Ollama, etc.).
- `populate_database.py` - Loads PDFs, splits into chunks, creates IDs and stores embeddings in Chroma.
- `query_data.py` - Runs a similarity search, formats a prompt and queries an LLM for an answer.
- `test_rag.py` - Small test harness that queries the system and uses an LLM to validate responses.

## Usage

1. Prepare data

   - Add one or more PDF files into the `data/` directory.

2. Populate the vector database

   - To create or update the Chroma DB:

     ```bash
     python populate_database.py
     ```

   - To reset (clear) the database and re-create it from scratch:

     ```bash
     python populate_database.py --reset
     ```

3. Query the database

   - Run a query from the command line:

     ```bash
     python query_data.py "How much total money does a player start with in Monopoly?"
     ```

   - The script will perform a similarity search, build a context prompt from the top results, and call the configured LLM.

4. Run tests

   - Execute the simple test file with pytest:

     ```bash
     pytest test_rag.py
     ```

## Configuration notes

- Embeddings / LLM configuration

  - Open `get_embedding_function.py` to choose which embedding implementation to use. The project currently uses `BedrockEmbeddings` by default. To use local Ollama embeddings, you can switch to `OllamaEmbeddings` instead.

  - `query_data.py` and `test_rag.py` use the `Ollama` model class from `langchain_community.llms.ollama` by default. Replace or adjust the model code if you want to use a different provider (e.g., an OpenAI or Bedrock LLM integration).

- Paths

  - The project uses `data/` for source PDFs and `chroma/` as the default ChromaDB persistence directory. These are defined as `DATA_PATH` and `CHROMA_PATH` in the scripts.

## Troubleshooting

- If embeddings or the LLM fail to connect, confirm your provider configuration (Ollama server running, or AWS credentials/profile for Bedrock).
- If you see no new documents being added, check that PDFs were placed under `data/` and that they contain extractable text.

## Contributing

This is a small tutorial repository — feel free to open issues or submit PRs to improve examples or add additional providers.

## License

This repository does not include an explicit license file. Add one if you plan to reuse or publish this project.
