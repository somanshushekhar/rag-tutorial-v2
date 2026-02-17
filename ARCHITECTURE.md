# 🏗️ Aflac Assist Chat - System Architecture

## 📋 Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Concepts](#core-concepts)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [Technology Stack](#technology-stack)
7. [API Endpoints](#api-endpoints)
8. [Database Schema](#database-schema)
9. [Security & Performance](#security--performance)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## 📖 Overview

### What is Aflac Assist Chat?

Aflac Assist Chat is a **Retrieval-Augmented Generation (RAG)** chatbot that enables users to:
- Upload PDF documents containing company policies, insurance information, etc.
- Ask natural language questions about the documents
- Receive AI-generated answers with source citations
- Experience real-time streaming responses

### Key Capabilities

✅ **Semantic Search**: Understands meaning, not just keywords  
✅ **Source Attribution**: Shows which documents were used  
✅ **Incremental Updates**: Add new documents without rebuilding  
✅ **Privacy-First**: Runs entirely locally using Ollama  
✅ **Real-time Streaming**: ChatGPT-like response experience  

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER BROWSER                            │
│  ┌──────────────┐                    ┌──────────────────────┐   │
│  │ Upload Page  │                    │   Chat Interface     │   │
│  │ (index.html) │                    │    (chat.html)       │   │
│  └──────┬───────┘                    └──────────┬───────────┘   │
└─────────┼──────────────────────────────────────┼───────────────┘
          │ HTTP POST                             │ HTTP POST
          │ (multipart/form-data)                 │ (Server-Sent Events)
          ▼                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI WEB SERVER                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        app.py                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐     │   │
│  │  │  /upload │  │  /query  │  │  /chat/stream      │     │   │
│  │  └─────┬────┘  └────┬─────┘  └──────────┬─────────┘     │   │
│  └────────┼────────────┼────────────────────┼───────────────┘   │
└───────────┼────────────┼────────────────────┼───────────────────┘
            │            │                    │
            ▼            ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                          │
│  ┌─────────────────────┐         ┌────────────────────────┐     │
│  │ populate_database.py│         │    query_data.py       │     │
│  │                     │         │                        │     │
│  │ • load_documents()  │         │ • query_rag()          │     │
│  │ • split_documents() │         │ • query_rag_streaming()│     │
│  │ • add_to_chroma()   │         │ • generate_with_ollama()│    │
│  └─────────┬───────────┘         └──────────┬─────────────┘     │
└─────────────┼──────────────────────────────┼───────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            get_embedding_function.py                     │   │
│  │                                                          │   │
│  │  class OllamaEmbeddingsWrapper:                          │   │
│  │    • embed_documents(texts) → vectors                    │   │
│  │    • embed_query(text) → vector                          │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│  ┌──────────────────┐              ┌─────────────────────────┐  │
│  │   ChromaDB       │              │    Ollama Server        │  │
│  │ (Vector Database)│              │  (localhost:11434)      │  │
│  │                  │              │                         │  │
│  │ • Stores vectors │              │ • nomic-embed-text      │  │
│  │ • Similarity     │              │   (embeddings)          │  │
│  │   search         │              │ • mistral/llama3.2      │  │
│  │ • Metadata       │              │   (text generation)     │  │
│  └──────────────────┘              └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Concepts

### 1. Retrieval-Augmented Generation (RAG)

**Traditional LLM Problem:**
- AI "hallucinates" facts
- No source attribution
- Limited to training data

**RAG Solution:**
```
User Question
    ↓
Retrieve relevant documents from your data
    ↓
Augment LLM prompt with retrieved context
    ↓
Generate answer based on facts
    ↓
Return answer + sources
```

### 2. Vector Embeddings

**What are they?**
Text converted to high-dimensional numerical vectors that capture semantic meaning.

**Example:**
```
"car accident"        → [0.23, 0.45, 0.67, ..., 0.12]  (1536 dimensions)
"vehicle collision"   → [0.24, 0.44, 0.68, ..., 0.11]  (very similar!)
"ice cream"           → [0.89, 0.02, 0.15, ..., 0.77]  (very different)
```

**Why important?**
- Enables semantic search (meaning-based)
- Finds "accident insurance" when user asks "coverage for injuries"
- Works across synonyms and paraphrases

### 3. Vector Similarity Search

**Cosine Similarity:**
Measures the angle between two vectors.
- 1.0 = identical
- 0.0 = completely different

**ChromaDB Process:**
```python
# 1. User query converted to vector
query_vector = [0.23, 0.45, 0.67, ...]

# 2. Compare with all stored document vectors
database = {
    "chunk_1": [0.24, 0.46, 0.68, ...],  # similarity: 0.98
    "chunk_2": [0.89, 0.02, 0.15, ...],  # similarity: 0.23
    "chunk_3": [0.22, 0.44, 0.66, ...],  # similarity: 0.99
}

# 3. Return top N most similar
results = ["chunk_3", "chunk_1"]  # Based on similarity scores
```

### 4. Document Chunking Strategy

**Why chunk?**
- LLMs have token limits (4K-32K)
- Smaller chunks = more precise retrieval
- Balance: too small loses context, too large loses precision

**Our Strategy:**
```
Chunk Size: 800 characters
Overlap: 80 characters

Document: "...AAABBBCCCDDD..."
         ↓
Chunk 1: "AAABBBCCC"
Chunk 2:     "BBBCCCDDD"  (overlap: "BBB")
```

**Why overlap?**
- Prevents splitting important information
- Maintains context across boundaries

### 5. Server-Sent Events (SSE)

**Real-time Streaming:**
```
Client: fetch('/chat/stream')
Server: data: {"type": "token", "content": "Based"}
        data: {"type": "token", "content": " on"}
        data: {"type": "token", "content": " the"}
        ...
        data: {"type": "sources", "content": [...]}
        data: [DONE]
```

**Advantages:**
- One-way server → client
- Built into browsers
- Simpler than WebSockets for this use case

---

## 🔧 Component Details

### 1. `get_embedding_function.py`

**Purpose:** Convert text to vectors using Ollama embeddings

**Class: OllamaEmbeddingsWrapper**

```python
class OllamaEmbeddingsWrapper:
    def __init__(self, model: str, base_url: str):
        self.model = model          # "nomic-embed-text"
        self.base_url = base_url    # "http://127.0.0.1:11434"
```

**Method: embed_documents**
```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """
    Input:  ["Hello world", "Another sentence"]
    Output: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    
    Process:
    1. POST to http://localhost:11434/api/embed
    2. Payload: {"model": "nomic-embed-text", "input": texts}
    3. Response: {"embeddings": [[...], [...]]}
    4. Return embeddings
    """
```

**Method: embed_query**
```python
def embed_query(self, text: str) -> List[float]:
    """
    Convenience method for single text.
    
    Input:  "What is covered?"
    Output: [0.123, 0.456, ...]
    """
    return self.embed_documents([text])[0]
```

**Usage:**
```python
# In populate_database.py
embeddings = embedding_fn.embed_documents(chunk_texts)

# In query_data.py
query_vector = embedding_fn.embed_query(user_question)
```

---

### 2. `populate_database.py`

**Purpose:** Process PDFs and build vector database

#### **Function: load_documents()**

```python
def load_documents() -> List[SimpleDocument]:
    """
    Loads all PDFs from data/ directory.
    
    Process:
    1. Scan data/ for *.pdf files
    2. For each PDF:
       - Read with PdfReader
       - Extract text from each page
       - Create document with metadata
    
    Returns:
    [
        SimpleDocument(
            page_content="Insurance policy text...",
            metadata={"source": "data/policy.pdf", "page": 1}
        ),
        ...
    ]
    """
```

**Data Structure:**
```python
@dataclass
class SimpleDocument:
    page_content: str     # Actual text content
    metadata: dict        # {"source": str, "page": int}
```

#### **Function: split_documents()**

```python
def split_documents(documents, chunk_size=800, chunk_overlap=80):
    """
    Breaks long documents into smaller chunks.
    
    Algorithm:
    for doc in documents:
        text = doc.page_content
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(SimpleDocument(chunk_text, doc.metadata))
            start = end - chunk_overlap  # Move back for overlap
    
    Example:
    Input:  "ABCDEFGHIJKLMNOPQRSTUVWXYZ" (chunk_size=10, overlap=2)
    Output: ["ABCDEFGHIJ", "IJKLMNOPQR", "QRSTUVWXYZ"]
                      ^^overlap      ^^overlap
    """
```

#### **Function: calculate_chunk_ids()**

```python
def calculate_chunk_ids(chunks):
    """
    Assigns unique IDs to prevent duplicates.
    
    ID Format: "{source}:{page}:{chunk_index}"
    
    Example:
    data/policy.pdf, page 3, chunk 0 → "data/policy.pdf:3:0"
    data/policy.pdf, page 3, chunk 1 → "data/policy.pdf:3:1"
    
    Why needed?
    - Same PDF uploaded twice won't create duplicates
    - Incremental updates work correctly
    """
```

#### **Function: add_to_chroma()**

```python
def add_to_chroma(chunks):
    """
    Stores chunks in ChromaDB with embeddings.
    
    Process:
    1. Connect to ChromaDB: client = chromadb.PersistentClient(path="chroma")
    2. Get or create collection: collection = client.get_or_create_collection("documents")
    3. Calculate chunk IDs
    4. Fetch existing IDs: existing_ids = set(collection.get()["ids"])
    5. Filter new chunks:
       for chunk in chunks:
           if chunk.id not in existing_ids:
               new_chunks.append(chunk)
    6. Generate embeddings: embeddings = embedding_fn.embed_documents(new_texts)
    7. Batch insert:
       for batch in batches(new_chunks, 256):
           collection.add(
               documents=batch_texts,
               metadatas=batch_metadatas,
               ids=batch_ids,
               embeddings=batch_embeddings
           )
    
    Result: Chunks stored with vectors for similarity search
    """
```

**Incremental Update Logic:**
```python
# Existing database: ["doc1.pdf:1:0", "doc1.pdf:1:1", ...]
# New upload: doc2.pdf
# Process:
# 1. doc2.pdf chunks: ["doc2.pdf:1:0", "doc2.pdf:1:1", ...]
# 2. Check: none exist in database
# 3. Add all doc2.pdf chunks
# 4. doc1.pdf chunks remain untouched
```

---

### 3. `query_data.py`

**Purpose:** Search documents and generate AI answers

#### **Function: query_rag()**

```python
def query_rag(query_text: str) -> str:
    """
    Synchronous RAG query for CLI usage.
    
    Steps:
    1. Get embedding function
    2. Connect to ChromaDB
    3. Convert query to vector
    4. Search for similar chunks
    5. Build context from results
    6. Create prompt with template
    7. Call Ollama LLM
    8. Return answer
    
    Input:  "What does Aflac cover?"
    Output: "Based on the documents, Aflac covers..."
    """
```

**Detailed Flow:**

```python
# STEP 1: Query Embedding
query_embedding = embedding_function.embed_query(query_text)
# "What does Aflac cover?" → [0.234, 0.567, ...]

# STEP 2: Vector Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,  # Top 5 most relevant
    include=["metadatas", "documents", "distances"]
)

# Results structure:
# {
#   "documents": [["chunk1 text", "chunk2 text", ...]],
#   "metadatas": [[{id, source, page}, {id, source, page}, ...]],
#   "distances": [[0.12, 0.15, 0.18, ...]]  # Lower = more similar
# }

# STEP 3: Build Context
docs = results["documents"][0]  # ["chunk1", "chunk2", ...]
context_text = "\n\n---\n\n".join(docs)

# STEP 4: Create Prompt
prompt = f"""
Answer the question based only on the following context:

{context_text}

---

Answer the question based on the above context: {query_text}
"""

# STEP 5: Call Ollama
response = generate_with_ollama(prompt, "mistral", "http://localhost:11434")

# STEP 6: Return
return response
```

#### **Function: query_rag_streaming()**

```python
async def query_rag_streaming(query_text: str):
    """
    Async generator for real-time streaming.
    
    Yields:
    {"type": "token", "content": "Based"}
    {"type": "token", "content": " on"}
    ...
    {"type": "sources", "content": [...]}
    {"type": "error", "content": "..."}  (if error)
    
    Usage:
    async for chunk in query_rag_streaming(query):
        if chunk["type"] == "token":
            print(chunk["content"], end="")
    """
```

**Streaming Implementation:**

```python
# Request with stream=True
resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "mistral", "prompt": prompt, "stream": True},
    stream=True
)

# Iterate over streaming response
for line in resp.iter_lines(decode_unicode=True):
    obj = json.loads(line)
    token = obj.get("response", "")
    if token:
        yield {"type": "token", "content": token}
    
    if obj.get("done"):
        break

# Send sources at end
yield {"type": "sources", "content": sources}
```

#### **Function: generate_with_ollama()**

```python
def generate_with_ollama(prompt: str, model: str, base_url: str) -> str:
    """
    Calls Ollama to generate text.
    
    Tries two methods:
    1. CLI (if ollama command available)
       subprocess.run(["ollama", "run", model, prompt])
    
    2. HTTP API (fallback)
       POST to /api/generate
    
    Returns complete response text.
    """
```

---

### 4. `app.py`

**Purpose:** FastAPI web server and routing

#### **Endpoint: GET /**

```python
@app.get("/")
def index(request: Request):
    """
    Renders upload page.
    
    Returns: templates/index.html
    """
```

#### **Endpoint: POST /upload**

```python
@app.post("/upload")
async def upload(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    reset: bool = Form(False)
):
    """
    Handles file uploads.
    
    Process:
    1. Receive multipart/form-data
    2. Save files to data/ directory
    3. Schedule background task: ingest_task(reset)
    4. Redirect to /?success=uploaded
    
    Background Task:
    - Runs populate_database functions
    - Doesn't block HTTP response
    - Logs progress to console
    """
```

**Background Task:**

```python
def ingest_task(reset: bool, filenames: List[str]):
    """
    Runs in background thread.
    
    Steps:
    1. If reset: clear_database()
    2. documents = load_documents()
    3. chunks = split_documents(documents)
    4. add_to_chroma(chunks)
    5. Log completion
    """
```

#### **Endpoint: POST /query**

```python
@app.post("/query")
async def query(request: Request, query: str = Form(...)):
    """
    Simple synchronous query (old interface).
    
    Process:
    1. Receive form data
    2. answer = query_rag(query)
    3. Render result.html with answer
    
    Returns: HTML page with answer
    """
```

#### **Endpoint: GET /chat**

```python
@app.get("/chat")
def chat_page(request: Request):
    """
    Renders chat interface.
    
    Returns: templates/chat.html
    """
```

#### **Endpoint: POST /chat/stream**

```python
@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Server-Sent Events endpoint for streaming.
    
    Process:
    1. Receive JSON: {"query": "..."}
    2. Create async generator
    3. Stream responses as SSE
    
    Response format:
    data: {"type": "token", "content": "text"}
    data: {"type": "sources", "content": [...]}
    data: [DONE]
    
    Content-Type: text/event-stream
    """
```

**SSE Implementation:**

```python
async def generate():
    async for chunk in query_rag_streaming(query_text):
        # Format as SSE
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # Allow other tasks
    
    yield "data: [DONE]\n\n"

return StreamingResponse(
    generate(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
)
```

---

### 5. Templates

#### **templates/index.html**

**Features:**
- PDF file input (multiple)
- Reset database checkbox
- Status messages (success/error)
- Link to chat interface
- JavaScript for URL parameter handling

**Form Submission:**
```html
<form action="/upload" enctype="multipart/form-data" method="post">
    <input type="file" name="files" accept=".pdf" multiple />
    <input type="checkbox" name="reset" value="true" />
    <button type="submit">Upload & Ingest</button>
</form>
```

#### **templates/chat.html**

**Features:**
- Modern chat UI
- Message bubbles (user/assistant)
- Typing indicator animation
- Real-time token streaming
- Source citations
- Auto-scroll

**Key JavaScript:**

```javascript
// Stream handling
const response = await fetch('/chat/stream', {
    method: 'POST',
    body: JSON.stringify({ query: userInput })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'token') {
                // Append token to message
                messageDiv.textContent += data.content;
            } else if (data.type === 'sources') {
                // Show sources
                displaySources(data.content);
            }
        }
    }
}
```

#### **templates/result.html**

**Features:**
- Simple answer display
- Back link to home
- Pre-formatted text

---

## 🔄 Data Flow

### Complete Upload → Query Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER UPLOADS insurance.pdf                               │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. FastAPI /upload endpoint                                 │
│    - Save to data/insurance.pdf                             │
│    - Start background task                                  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. populate_database.py                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ load_documents()                                 │    │
│    │   Input: data/insurance.pdf                      │    │
│    │   Output: [Document(page1), Document(page2), ...] │  │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ split_documents()                                │    │
│    │   Input: 3 pages (10,000 chars total)            │    │
│    │   Output: 15 chunks (800 chars each)             │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ calculate_chunk_ids()                            │    │
│    │   Assign IDs: "insurance.pdf:1:0", ":1:1", ...   │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ add_to_chroma()                                  │    │
│    │   1. Get embeddings from Ollama                  │    │
│    │      ["chunk1",...] → [[0.1,...], [0.2,...],...]  │   │
│    │   2. Store in ChromaDB                           │    │
│    │      {id, text, vector, metadata}                │    │
│    └──────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ChromaDB Storage                                         │
│    Collection: "documents"                                  │
│    ┌────────────────────────────────────────────────────┐  │
│    │ ID: "insurance.pdf:1:0"                            │  │
│    │ Text: "Aflac provides accident insurance..."       │  │
│    │ Vector: [0.234, 0.567, 0.123, ...]                 │  │
│    │ Metadata: {source: "...", page: 1}                 │  │
│    └────────────────────────────────────────────────────┘  │
│    ... (14 more chunks)                                    │
└─────────────────────────────────────────────────────────────┘

        TIME PASSES... User visits chat interface

┌─────────────────────────────────────────────────────────────┐
│ 5. USER ASKS: "What does Aflac cover?"                      │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. FastAPI /chat/stream endpoint                            │
│    - Parse JSON request                                     │
│    - Call query_rag_streaming()                             │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. query_rag_streaming()                                    │
│    ┌──────────────────────────────────────────────────┐    │
│    │ STEP 1: Embed query                              │    │
│    │   "What does Aflac cover?"                        │    │
│    │   → Ollama embedding API                          │    │
│    │   → [0.245, 0.583, 0.129, ...]                    │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ STEP 2: ChromaDB vector search                   │    │
│    │   Query: [0.245, 0.583, ...]                      │    │
│    │   Search: Compare with all stored vectors        │    │
│    │   Results (top 5):                               │    │
│    │     1. "insurance.pdf:5:2" (distance: 0.08)       │    │
│    │     2. "insurance.pdf:12:1" (distance: 0.12)      │    │
│    │     3. "insurance.pdf:3:0" (distance: 0.15)       │    │
│    │     4. "insurance.pdf:8:3" (distance: 0.18)       │    │
│    │     5. "insurance.pdf:1:1" (distance: 0.21)       │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ STEP 3: Build context                            │    │
│    │   context = "Aflac provides accident insurance... │    │
│    │              ---                                  │    │
│    │              Hospital coverage includes...        │    │
│    │              ---                                  │    │
│    │              Cancer insurance pays..."            │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ STEP 4: Create prompt                            │    │
│    │   "Answer based only on the following context:   │    │
│    │    [context]                                      │    │
│    │    ---                                            │    │
│    │    Question: What does Aflac cover?"             │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ STEP 5: Stream from Ollama                       │    │
│    │   POST /api/generate (stream=true)               │    │
│    │   Response stream:                               │    │
│    │     {"response": "Based", "done": false}          │    │
│    │     {"response": " on", "done": false}            │    │
│    │     {"response": " the", "done": false}           │    │
│    │     {"response": " documents", "done": false}     │    │
│    │     ...                                           │    │
│    └──────────────────┬───────────────────────────────┘    │
│                       ▼                                     │
│    ┌──────────────────────────────────────────────────┐    │
│    │ STEP 6: Yield tokens                             │    │
│    │   yield {"type": "token", "content": "Based"}     │    │
│    │   yield {"type": "token", "content": " on"}       │    │
│    │   ...                                             │    │
│    │   yield {"type": "sources", "content": [...]}     │    │
│    └──────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. FastAPI streams to browser                               │
│    data: {"type": "token", "content": "Based"}              │
│    data: {"type": "token", "content": " on"}                │
│    ...                                                      │
│    data: {"type": "sources", "content": [...]}              │
│    data: [DONE]                                             │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Browser JavaScript displays                              │
│    - Append each token to message bubble                    │
│    - Show typing effect                                     │
│    - Display sources at end                                 │
│    - Enable input for next question                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

### Backend Technologies

| Technology | Version | Purpose | Why Chosen |
|------------|---------|---------|------------|
| **Python** | 3.11+ | Programming language | Best AI/ML ecosystem |
| **FastAPI** | 0.110.0 | Web framework | Fast, async, auto docs |
| **Uvicorn** | 0.30.0+ | ASGI server | Production-ready async |
| **ChromaDB** | 0.4.15 | Vector database | Easy, embedded, fast |
| **pypdf** | 3.17.0 | PDF parsing | Simple, reliable |
| **Requests** | 2.32.3 | HTTP client | Ollama API calls |

### AI/ML Technologies

| Technology | Model | Purpose |
|------------|-------|---------|
| **Ollama** | nomic-embed-text | Text embeddings (1536 dims) |
| **Ollama** | mistral / llama3.2 | Text generation |

### Frontend Technologies

| Technology | Purpose |
|------------|---------|
| **HTML5** | Structure |
| **CSS3** | Styling (gradients, animations) |
| **Vanilla JavaScript** | Interactivity, SSE handling |
| **Fetch API** | HTTP requests |
| **Streams API** | Real-time response handling |

### Data Storage

| Storage | Purpose | Location |
|---------|---------|----------|
| **ChromaDB** | Vector embeddings + metadata | `chroma/` directory |
| **File System** | PDF documents | `data/` directory |

---

## 🔌 API Endpoints

### Frontend Routes

#### `GET /`
**Purpose:** Upload interface  
**Returns:** `templates/index.html`  
**Query Params:**
- `?success=uploaded` - Show success message
- `?error=upload_failed` - Show error message

#### `GET /chat`
**Purpose:** Chat interface  
**Returns:** `templates/chat.html`

---

### Backend API Routes

#### `POST /upload`
**Purpose:** Upload PDF files

**Request:**
```http
POST /upload
Content-Type: multipart/form-data

files: [File, File, ...]
reset: "true" | undefined
```

**Response:**
```http
HTTP/1.1 303 See Other
Location: /?success=uploaded
```

**Process:**
1. Save files to `data/`
2. Schedule background ingestion
3. Redirect with status

---

#### `POST /query`
**Purpose:** Simple synchronous query

**Request:**
```http
POST /query
Content-Type: application/x-www-form-urlencoded

query=What+does+Aflac+cover%3F
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: text/html

<html>
  <h1>Query Result</h1>
  <p>Based on the documents...</p>
</html>
```

---

#### `POST /chat/stream`
**Purpose:** Streaming chat responses

**Request:**
```http
POST /chat/stream
Content-Type: application/json

{
  "query": "What does Aflac cover?"
}
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"type": "token", "content": "Based"}

data: {"type": "token", "content": " on"}

data: {"type": "token", "content": " the"}

data: {"type": "sources", "content": [{"rank": 1, "source": "...", "distance": 0.08}]}

data: [DONE]

```

**SSE Message Types:**

| Type | Content | Purpose |
|------|---------|---------|
| `token` | String | Single word/token from LLM |
| `sources` | Array | Source documents used |
| `error` | String | Error message |

---

## 🗄️ Database Schema

### ChromaDB Collection: "documents"

**Storage Format:**
```python
{
    "ids": ["insurance.pdf:1:0", "insurance.pdf:1:1", ...],
    "documents": ["Aflac provides...", "Coverage includes...", ...],
    "embeddings": [[0.234, 0.567, ...], [0.123, 0.456, ...], ...],
    "metadatas": [
        {"source": "data/insurance.pdf", "page": 1, "id": "insurance.pdf:1:0"},
        {"source": "data/insurance.pdf", "page": 1, "id": "insurance.pdf:1:1"},
        ...
    ]
}
```

**Field Descriptions:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `ids` | string | Unique identifier | `"policy.pdf:5:2"` |
| `documents` | string | Chunk text content | `"Aflac offers accident..."` |
| `embeddings` | float[] | 1536-dim vector | `[0.234, 0.567, ...]` |
| `metadatas.source` | string | Original file path | `"data/policy.pdf"` |
| `metadatas.page` | int | Page number | `5` |
| `metadatas.id` | string | Same as ids | `"policy.pdf:5:2"` |

**Indexing:**
- Vectors indexed using HNSW (Hierarchical Navigable Small World)
- Approximate nearest neighbor search
- O(log n) query time

---

## 🔒 Security & Performance

### Security Considerations

#### Current Implementation
✅ **Runs locally** - No data sent to external services  
✅ **No authentication required** - Internal use  
⚠️ **No file validation** - Accepts any .pdf  
⚠️ **No rate limiting** - Can be overwhelmed  

#### Production Recommendations

1. **File Validation:**
```python
# Add to upload endpoint
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = ["application/pdf"]

if file.size > MAX_FILE_SIZE:
    raise HTTPException(413, "File too large")

if file.content_type not in ALLOWED_TYPES:
    raise HTTPException(415, "Invalid file type")
```

2. **Authentication:**
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

@app.post("/upload")
async def upload(credentials: HTTPBasicCredentials = Depends(security)):
    # Verify credentials
    if not verify_user(credentials):
        raise HTTPException(401, "Invalid credentials")
```

3. **Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request):
    ...
```

### Performance Optimization

#### Current Performance
- **Upload**: ~1-5 seconds per MB (depends on PDF complexity)
- **Embedding**: ~100ms per chunk
- **Query**: ~200-500ms total
- **Streaming**: ~50-100 tokens/second

#### Optimization Strategies

1. **Batch Embedding:**
```python
# Instead of:
for chunk in chunks:
    embedding = embed([chunk])[0]

# Use:
embeddings = embed(chunks)  # Single API call
```

2. **Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_embedding(text: str):
    return embedding_fn.embed_query(text)
```

3. **Parallel Processing:**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    embeddings = list(executor.map(embed_single, chunks))
```

4. **Connection Pooling:**
```python
import requests
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_maxsize=10)
session.mount('http://', adapter)
```

---

## 🚀 Deployment

### Local Development

```bash
# 1. Start Ollama
ollama serve

# 2. Activate venv
.venv\Scripts\Activate.ps1

# 3. Run app
python -m uvicorn app:app --reload --port 8000
```

### Production Deployment Options

#### Option 1: Docker Container

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./chroma:/app/chroma
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

#### Option 2: Linux Server (systemd)

```ini
# /etc/systemd/system/aflac-assist.service
[Unit]
Description=Aflac Assist Chat
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/aflac-assist
Environment="PATH=/opt/aflac-assist/.venv/bin"
ExecStart=/opt/aflac-assist/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable aflac-assist
sudo systemctl start aflac-assist
```

#### Option 3: Cloud (Azure/AWS)

**Azure App Service:**
```bash
az webapp up --name aflac-assist --runtime "PYTHON:3.11"
```

**AWS EC2:**
```bash
# User data script
#!/bin/bash
git clone https://github.com/somanshushekhar/Aflac-Assist-Chat.git
cd Aflac-Assist-Chat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error:**
```
ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 11434)
```

**Solution:**
```bash
# Find process using port
netstat -ano | findstr :11434

# Kill process
taskkill /PID <PID> /F

# Or use different port
uvicorn app:app --port 8001
```

#### 2. Ollama Connection Failed

**Error:**
```
ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded
```

**Solution:**
```bash
# Check Ollama is running
ollama list

# If not running
ollama serve

# Verify endpoint
curl http://localhost:11434/api/tags
```

#### 3. ChromaDB Initialization Error

**Error:**
```
ValueError: Could not connect to ChromaDB
```

**Solution:**
```bash
# Delete corrupted database
rm -rf chroma/

# Restart app (will recreate)
python -m uvicorn app:app --reload
```

#### 4. Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce batch size in populate_database.py
batch_size = 128  # Instead of 256
```

#### 5. Slow Embeddings

**Issue:** Taking too long to process documents

**Solutions:**
```bash
# 1. Use faster model
export OLLAMA_EMBED_MODEL="all-minilm"

# 2. Reduce chunk size
# In populate_database.py:
chunk_size = 500  # Instead of 800

# 3. Use GPU acceleration (if available)
ollama run nomic-embed-text --gpu
```

---

## 📊 Monitoring & Logging

### Current Logging

```python
# app.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("📖 Loading documents...")
logger.error("❌ Error during ingestion: {e}")
```

### Production Logging

```python
# logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "aflac-assist.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
```

---

## 🎓 Learning Resources

### RAG Concepts
- [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Embeddings Explained](https://www.pinecone.io/learn/vector-embeddings/)

### Technologies
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)

### Advanced Topics
- [Semantic Search](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

## 📝 Appendix

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_MODEL` | `mistral` | LLM for generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### File Structure

```
Aflac-Assist-Chat/
├── .venv/                  # Virtual environment
├── chroma/                 # ChromaDB storage (gitignored)
├── data/                   # PDF uploads (gitignored)
├── templates/
│   ├── index.html         # Upload page
│   ├── chat.html          # Chat interface
│   └── result.html        # Query result page
├── .gitignore
├── app.py                 # FastAPI application
├── get_embedding_function.py
├── populate_database.py
├── query_data.py
├── requirements.txt
├── test_setup.py
├── test_rag.py
├── README.md
└── ARCHITECTURE.md        # This file
```

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Somansh Shekhar  
**Project:** Aflac Assist Chat  
**License:** MIT
