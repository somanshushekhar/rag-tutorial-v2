from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List
import logging
import json
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse functions from your scripts
from populate_database import load_documents, split_documents, add_to_chroma, clear_database, DATA_PATH
from query_data import query_rag

DATA_DIR = Path(DATA_PATH)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Optional: serve a static directory (e.g., CSS) if it exists
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


def ingest_task(reset: bool = False, filenames: List[str] = None):
    """
    Background ingestion task that optionally clears the DB then ingests all PDFs in `data/`.
    Runs synchronously in the background (FastAPI BackgroundTasks calls this in a worker thread).
    """
    try:
        if reset:
            logger.info("🗑️  Clearing database...")
            clear_database()
            logger.info("✅ Database cleared")

        logger.info("📖 Loading documents...")
        documents = load_documents()
        logger.info(f"✅ Loaded {len(documents)} document pages")

        logger.info("✂️  Splitting documents into chunks...")
        chunks = split_documents(documents)
        logger.info(f"✅ Created {len(chunks)} chunks")

        logger.info("💾 Adding to ChromaDB...")
        add_to_chroma(chunks)
        logger.info("✅ Ingestion complete!")

        if filenames:
            logger.info(f"📄 Processed files: {', '.join(filenames)}")
    except Exception as e:
        logger.error(f"❌ Error during ingestion: {e}")
        raise


@app.get("/")
def index(request: Request):
    """
    Render the main HTML page with upload and query forms.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),  # ensure FastAPI injects a BackgroundTasks instance
    reset: bool = Form(False),
):
    """
    Upload one or more PDF files and start background ingestion.
    If `reset` is provided (form), the Chroma DB will be cleared before ingestion.

    NOTE: FastAPI will inject a BackgroundTasks instance. We keep a default to avoid None issues
    (BackgroundTasks is lightweight).
    """
    # Save uploaded files
    uploaded_filenames = []
    for upload_file in files:
        dest_path = DATA_DIR / upload_file.filename
        try:
            content = await upload_file.read()
            dest_path.write_bytes(content)
            uploaded_filenames.append(upload_file.filename)
            logger.info(f"📁 Saved: {upload_file.filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save {upload_file.filename}: {e}")
            return RedirectResponse(url="/?error=upload_failed", status_code=303)

    # Run ingestion in background (non-blocking)
    try:
        background_tasks.add_task(ingest_task, reset, uploaded_filenames)
        logger.info(f"🚀 Started background ingestion for {len(uploaded_filenames)} file(s)")
    except Exception:
        # fallback: run synchronously if background scheduling fails
        logger.warning("⚠️  Background task failed, running synchronously")
        ingest_task(reset, uploaded_filenames)

    return RedirectResponse(url="/?success=uploaded", status_code=303)


@app.post("/query")
async def query(request: Request, query: str = Form(...)):
    """
    Run a synchronous query and render the result page.
    Uses your existing `query_rag` function (returns the raw LLM response string).
    """
    # Call existing pipeline (this runs synchronously; for long queries consider moving to background + polling)
    answer = query_rag(query)
    answer_text = answer if isinstance(answer, str) else ""
    return templates.TemplateResponse("result.html", {"request": request, "query": query, "answer": answer_text})


@app.get("/chat")
def chat_page(request: Request):
    """
    Render the chat interface page.
    """
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Stream chat responses using Server-Sent Events (SSE).
    """
    body = await request.json()
    query_text = body.get("query", "")

    if not query_text:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'content': 'No query provided'})}\n\n"]),
            media_type="text/event-stream"
        )

    async def generate():
        try:
            # Import here to avoid circular imports
            from query_data import query_rag_streaming

            # Stream the response
            async for chunk in query_rag_streaming(query_text):
                if chunk["type"] == "error":
                    yield f"data: {json.dumps(chunk)}\n\n"
                    break
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
