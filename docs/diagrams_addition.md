

```javascript
// Add to chat.html <script> section

const STORAGE_KEY = 'aflac_chat_history';
const MAX_MESSAGES = 100; // Prevent storage overflow

// Load on page load
window.addEventListener('DOMContentLoaded', () => {
  loadChatHistory();
});

function loadChatHistory() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const history = JSON.parse(saved);
      messages = history.slice(-MAX_MESSAGES);
      renderAllMessages();
      console.log(`✓ Loaded ${messages.length} messages from history`);
    }
  } catch (error) {
    console.error('Failed to load chat history:', error);
    messages = [];
  }
}

function saveChatHistory() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch (error) {
    if (error.name === 'QuotaExceededError') {
      // Keep only recent messages if quota exceeded
      messages = messages.slice(-50);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    }
  }
}

function clearChatHistory() {
  if (confirm('Clear all chat history?')) {
    localStorage.removeItem(STORAGE_KEY);
    messages = [];
    renderAllMessages();
  }
}

// Call after adding each message
function addMessage(role, content, sources = null) {
  messages.push({ role, content, sources, timestamp: new Date().toISOString() });
  saveChatHistory(); // ← Save to localStorage
  renderMessage(messages[messages.length - 1]);
}
```

### 11.3 Comparison of Persistence Options

```
┌──────────────────────────────────────────────────────────────────┐
│                    PERSISTENCE COMPARISON                        │
└──────────────────────────────────────────────────────────────────┘

Feature              │ None     │ LocalStorage │ SessionStorage │ Database
─────────────────────┼──────────┼──────────────┼────────────────┼──────────
Persist on Refresh   │ ❌       │ ✅           │ ✅             │ ✅
Persist on Close     │ ❌       │ ✅           │ ❌             │ ✅
Cross-Device         │ ❌       │ ❌           │ ❌             │ ✅
Backend Required     │ ❌       │ ❌           │ ❌             │ ✅
Implementation Time  │ 0 min    │ 30 min       │ 30 min         │ 4-8 hrs
Storage Limit        │ N/A      │ 5-10 MB      │ 5-10 MB        │ Unlimited
Privacy              │ ✅ Best  │ ✅ Local     │ ✅ Local       │ ⚠️ Server
User Auth Needed     │ ❌       │ ❌           │ ❌             │ ✅
Search History       │ ❌       │ ⚠️ Client    │ ❌             │ ✅
Export/Import        │ ❌       │ ⚠️ Manual    │ ❌             │ ✅
Multi-User Support   │ ❌       │ ❌           │ ❌             │ ✅
```

**Recommendation:** Start with **LocalStorage** (quick win), then upgrade to **Database** for production.

---

## 12. Error Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING FLOW                           │
└─────────────────────────────────────────────────────────────────┘

                         User Request
                              │
                              ▼
                    ┌─────────────────┐
                    │  FastAPI Route  │
                    └────────┬────────┘
                             │
              ╔══════════════╧══════════════╗
              ║     Try-Except Block        ║
              ╚══════════════╤══════════════╝
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐      ┌──────────────┐     ┌──────────┐
    │ Success │      │ Known Error  │     │ Unknown  │
    └────┬────┘      └──────┬───────┘     │  Error   │
         │                  │              └────┬─────┘
         ▼                  ▼                   ▼
    Return 200         Return 4xx/5xx      Return 500


ERROR TYPES:

1️⃣  ConnectionError - Ollama not running
   ┌────────────────────────────────────────────────────────────┐
   │ HTTP 503 Service Unavailable                              │
   │ Message: "AI service unavailable. Start Ollama server."    │
   │ Action: Run 'ollama serve'                                 │
   └────────────────────────────────────────────────────────────┘

2️⃣  EmptyDatabaseError - No documents
   ┌────────────────────────────────────────────────────────────┐
   │ HTTP 400 Bad Request                                       │
   │ Message: "No documents found. Upload PDFs first."          │
   │ Action: Go to upload page                                  │
   └────────────────────────────────────────────────────────────┘

3️⃣  InvalidFileError - Bad PDF
   ┌────────────────────────────────────────────────────────────┐
   │ HTTP 415 Unsupported Media Type                            │
   │ Message: "Invalid PDF file."                               │
   │ Action: Check file and retry                               │
   └────────────────────────────────────────────────────────────┘

4️⃣  TimeoutError - Query too slow
   ┌────────────────────────────────────────────────────────────┐
   │ HTTP 504 Gateway Timeout                                   │
   │ Message: "Request timed out. Try simpler question."        │
   │ Action: Rephrase query                                     │
   └────────────────────────────────────────────────────────────┘

5️⃣  ModelNotFoundError - Missing Ollama model
   ┌────────────────────────────────────────────────────────────┐
   │ HTTP 500 Internal Server Error                             │
   │ Message: "AI model not found. Run: ollama pull mistral"    │
   │ Action: Install model                                      │
   └────────────────────────────────────────────────────────────┘
```

---

## 13. Database Entity Relationship

```
┌──────────────────────────────────────────────────────────────────┐
│                      CHROMADB STRUCTURE                          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                 PersistentClient (./chroma/)                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Contains
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              Collection: "documents"                             │
│              Embedding: OllamaEmbeddings (1536-dim)              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │   IDs    │   │Documents │   │Embeddings│
       └────┬─────┘   └────┬─────┘   └────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     DOCUMENT RECORD                              │
├──────────────────────────────────────────────────────────────────┤
│ ID: "data/insurance.pdf:5:2"                                    │
│   ├─ Source: "data/insurance.pdf"                               │
│   ├─ Page: 5                                                    │
│   └─ Chunk: 2                                                   │
│                                                                  │
│ Document: "Aflac provides accident insurance..." (800 chars)    │
│                                                                  │
│ Embedding: [0.234, 0.567, -0.123, ...] (1536 floats)            │
│                                                                  │
│ Metadata: {source: "data/insurance.pdf", page: 5, id: "..."}    │
└──────────────────────────────────────────────────────────────────┘

RELATIONSHIPS:

     One PDF File
          │
          │ contains
          ▼
    Many Pages (1-N)
          │
          │ split into
          ▼
    Many Chunks (1-N)
          │
          │ stored as
          ▼
   Vector Records (1-1)

Example: insurance.pdf
  ├─ Page 1 → 3 chunks → 3 records
  ├─ Page 2 → 2 chunks → 2 records  
  ├─ Page 3 → 4 chunks → 4 records
  └─ Total: 9 vector records
```

---

## 14. Summary: Quick Reference

| Diagram | Key Question Answered | Section |
|---------|----------------------|---------|
| System Architecture | How does everything fit together? | 1 |
| Component Interaction | How do parts communicate? | 2 |
| Sequence Diagrams | What happens step-by-step? | 3 |
| Data Flow | How is data transformed? | 4 |
| Deployment | How do I host this? | 5 |
| Network Flow | What is the request path? | 6 |
| State Machine | What are possible app states? | 7 |
| Technology Stack | What technologies are used? | 8 |
| Security | What are security considerations? | 9 |
| Performance | How fast is it? | 10 |
| **Chat History** | **Why is my history lost?** | **11** |
| Error Handling | What errors can occur? | 12 |
| Database ERD | How is data structured? | 13 |

---

**Quick Start Guide:**
1. **New to the project?** → Read System Architecture (Section 1)
2. **Want to understand RAG?** → See Data Flow Diagrams (Section 4)
3. **Need to deploy?** → Check Deployment Architecture (Section 5)
4. **Troubleshooting?** → Error Flow Diagram (Section 12)
5. **Chat history issue?** → Chat History State (Section 11)

---

**Document Version:** 2.0  
**Last Updated:** 2024  
**Author:** Somansh Shekhar  
**Project:** Aflac Assist Chat  
**License:** MIT
