# 🏗️ AWS Deployment Architecture for Aflac Assist Chat

## 📊 Architecture Diagrams

### **1. Single EC2 Instance Architecture** (Recommended for Start)

```
                                    INTERNET
                                       │
                                       │ HTTPS/HTTP
                                       ▼
                    ┌──────────────────────────────────┐
                    │   AWS Route 53 (Optional)        │
                    │   Domain: aflac-assist.com       │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  Elastic IP (Static IP)          │
                    │  54.123.45.67                    │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────┐
│  AWS EC2 Instance (t3.xlarge or g4dn.xlarge)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Security Group                                          │  │
│  │  • Port 22 (SSH) - Your IP only                          │  │
│  │  • Port 80 (HTTP) - 0.0.0.0/0                            │  │
│  │  • Port 443 (HTTPS) - 0.0.0.0/0                          │  │
│  │  • Port 8000 (FastAPI) - 0.0.0.0/0                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Nginx Reverse Proxy (Port 80/443)                       │  │
│  │  • SSL/TLS Termination                                   │  │
│  │  • Proxy to FastAPI (Port 8000)                          │  │
│  └──────────────┬───────────────────────────────────────────┘  │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (uvicorn)                           │  │
│  │  Port: 8000                                              │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐ │  │
│  │  │  app.py                                            │ │  │
│  │  │  • /upload (POST)                                  │ │  │
│  │  │  • /chat (GET)                                     │ │  │
│  │  │  • /chat/stream (POST - SSE)                       │ │  │
│  │  │  • / (GET - Upload page)                           │ │  │
│  │  └────────────────────────────────────────────────────┘ │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐ │  │
│  │  │  populate_database.py                              │ │  │
│  │  │  • load_documents()                                │ │  │
│  │  │  • split_documents()                               │ │  │
│  │  │  • add_to_chroma()                                 │ │  │
│  │  └────────────────────────────────────────────────────┘ │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐ │  │
│  │  │  query_data.py                                     │ │  │
│  │  │  • query_rag_streaming()                           │ │  │
│  │  │  • Vector search                                   │ │  │
│  │  └────────────────────────────────────────────────────┘ │  │
│  └───────────┬──────────────────────────────────────────────┘  │
│              │                                                  │
│              ▼                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Ollama Server                                           │  │
│  │  Port: 11434                                             │  │
│  │                                                          │  │
│  │  Models:                                                 │  │
│  │  • nomic-embed-text (Embeddings - 1536 dims)            │  │
│  │  • mistral (Text Generation)                            │  │
│  │                                                          │  │
│  │  API Endpoints:                                          │  │
│  │  • POST /api/embed                                       │  │
│  │  • POST /api/generate                                    │  │
│  └───────────┬──────────────────────────────────────────────┘  │
│              │                                                  │
│              ▼                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ChromaDB (Vector Database)                              │  │
│  │  Path: /opt/aflac-assist/chroma/                         │  │
│  │                                                          │  │
│  │  Collection: "documents"                                 │  │
│  │  • Text chunks                                           │  │
│  │  • Embeddings (vectors)                                  │  │
│  │  • Metadata (source, page, id)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
          ┌────────────────────────────┐
          │  Amazon EBS Volume         │
          │  Type: gp3                 │
          │  Size: 100 GB              │
          │                            │
          │  /opt/aflac-assist/data/   │
          │  • PDF files               │
          │                            │
          │  /opt/aflac-assist/chroma/ │
          │  • Vector database         │
          └────────────────────────────┘
```

---

### **2. Production Architecture** (High Availability)

```
                                INTERNET
                                   │
                                   │ HTTPS
                                   ▼
                    ┌──────────────────────────────┐
                    │  AWS Route 53                │
                    │  aflac-assist.yourdomain.com │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  AWS Certificate Manager     │
                    │  SSL/TLS Certificate         │
                    └──────────┬───────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │  Application Load Balancer (ALB)                 │
        │  • Health checks                                 │
        │  • SSL termination                               │
        │  • Auto-scaling integration                      │
        └────────┬──────────────────────────┬──────────────┘
                 │                          │
                 ▼                          ▼
    ┌─────────────────────────┐  ┌─────────────────────────┐
    │  Availability Zone 1    │  │  Availability Zone 2    │
    │  ┌───────────────────┐  │  │  ┌───────────────────┐  │
    │  │  EC2 Instance 1   │  │  │  │  EC2 Instance 2   │  │
    │  │  • FastAPI        │  │  │  │  • FastAPI        │  │
    │  │  • Ollama         │  │  │  │  • Ollama         │  │
    │  └─────────┬─────────┘  │  │  └─────────┬─────────┘  │
    └────────────┼────────────┘  └────────────┼────────────┘
                 │                            │
                 └────────────┬───────────────┘
                              │
                              ▼
                   ┌────────────────────────┐
                   │  Amazon EFS            │
                   │  (Shared File System)  │
                   │  • PDF files           │
                   │  • ChromaDB data       │
                   └────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────┐
                   │  S3 Bucket (Backup)    │
                   │  • Daily backups       │
                   │  • Versioning enabled  │
                   └────────────────────────┘
```

---

### **3. Data Flow Diagram**

```
┌──────────────┐
│  User        │
│  Browser     │
└──────┬───────┘
       │
       │ 1. Upload PDF
       ▼
┌──────────────────────┐
│  FastAPI /upload     │
└──────┬───────────────┘
       │
       │ 2. Save to /data/
       ▼
┌──────────────────────┐
│  populate_database.py│
└──────┬───────────────┘
       │
       │ 3. Extract text
       ▼
┌──────────────────────┐
│  PdfReader           │
│  • Parse PDF         │
│  • Extract pages     │
└──────┬───────────────┘
       │
       │ 4. Split into chunks (800 chars)
       ▼
┌──────────────────────┐
│  split_documents()   │
│  • Chunk size: 800   │
│  • Overlap: 80       │
└──────┬───────────────┘
       │
       │ 5. Generate embeddings
       ▼
┌──────────────────────┐
│  Ollama API          │
│  POST /api/embed     │
│  Model: nomic-embed  │
└──────┬───────────────┘
       │
       │ 6. Returns 1536-dim vectors
       ▼
┌──────────────────────┐
│  ChromaDB            │
│  collection.add()    │
│  • Text chunks       │
│  • Embeddings        │
│  • Metadata          │
└──────────────────────┘

─── TIME PASSES ───

┌──────────────┐
│  User asks   │
│  question    │
└──────┬───────┘
       │
       │ 7. POST /chat/stream
       ▼
┌──────────────────────┐
│  query_data.py       │
└──────┬───────────────┘
       │
       │ 8. Embed query
       ▼
┌──────────────────────┐
│  Ollama API          │
│  POST /api/embed     │
└──────┬───────────────┘
       │
       │ 9. Query vector: [0.234, ...]
       ▼
┌──────────────────────┐
│  ChromaDB            │
│  similarity_search() │
└──────┬───────────────┘
       │
       │ 10. Top 5 relevant chunks
       ▼
┌──────────────────────┐
│  Build context       │
│  Create prompt       │
└──────┬───────────────┘
       │
       │ 11. POST /api/generate (stream=true)
       ▼
┌──────────────────────┐
│  Ollama API          │
│  Model: mistral      │
└──────┬───────────────┘
       │
       │ 12. Stream tokens
       ▼
┌──────────────────────┐
│  FastAPI SSE         │
│  data: {"type":      │
│  "token", ...}       │
└──────┬───────────────┘
       │
       │ 13. Server-Sent Events
       ▼
┌──────────────────────┐
│  Browser JavaScript  │
│  • Display tokens    │
│  • Show sources      │
└──────────────────────┘
```

---

### **4. AWS Resource Map**

```
┌─────────────────────────────────────────────────────────┐
│  AWS Account                                            │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Region: us-east-1 (or your choice)            │    │
│  │                                                 │    │
│  │  ┌──────────────────────────────────────────┐  │    │
│  │  │  VPC (Default or Custom)                 │  │    │
│  │  │                                          │  │    │
│  │  │  ┌────────────────────────────────────┐  │  │    │
│  │  │  │  Security Group                    │  │  │    │
│  │  │  │  Name: aflac-assist-sg             │  │  │    │
│  │  │  │  Inbound:                          │  │  │    │
│  │  │  │  • 22 (SSH)                        │  │  │    │
│  │  │  │  • 80 (HTTP)                       │  │  │    │
│  │  │  │  • 443 (HTTPS)                     │  │  │    │
│  │  │  │  • 8000 (FastAPI)                  │  │  │    │
│  │  │  └────────────────────────────────────┘  │  │    │
│  │  │                                          │  │    │
│  │  │  ┌────────────────────────────────────┐  │  │    │
│  │  │  │  EC2 Instance                      │  │  │    │
│  │  │  │  Name: Aflac-Assist-Chat           │  │  │    │
│  │  │  │  Type: t3.xlarge                   │  │  │    │
│  │  │  │  OS: Ubuntu 22.04 LTS              │  │  │    │
│  │  │  │  vCPU: 4                           │  │  │    │
│  │  │  │  RAM: 16 GB                        │  │  │    │
│  │  │  └────────┬───────────────────────────┘  │  │    │
│  │  │           │                              │  │    │
│  │  │           ▼                              │  │    │
│  │  │  ┌────────────────────────────────────┐  │  │    │
│  │  │  │  EBS Volume                        │  │  │    │
│  │  │  │  Type: gp3                         │  │  │    │
│  │  │  │  Size: 100 GB                      │  │  │    │
│  │  │  │  IOPS: 3000                        │  │  │    │
│  │  │  │  Throughput: 125 MB/s              │  │  │    │
│  │  │  └────────────────────────────────────┘  │  │    │
│  │  │                                          │  │    │
│  │  │  ┌────────────────────────────────────┐  │  │    │
│  │  │  │  Elastic IP (Optional)             │  │  │    │
│  │  │  │  Static Public IP                  │  │  │    │
│  │  │  └────────────────────────────────────┘  │  │    │
│  │  └──────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  IAM (Optional)                                 │    │
│  │  • EC2 instance role                            │    │
│  │  • S3 access for backups                        │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  S3 Bucket (Optional)                           │    │
│  │  Name: aflac-assist-backups                     │    │
│  │  • ChromaDB backups                             │    │
│  │  • Application logs                             │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Route 53 (Optional)                            │    │
│  │  Domain: aflac-assist.yourdomain.com            │    │
│  │  Record: A → Elastic IP                         │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Certificate Manager (Optional)                 │    │
│  │  SSL Certificate for HTTPS                      │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

### **5. Component Communication Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│  EC2 Instance (10.0.1.100)                                  │
│                                                             │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │   Nginx     │ ◄─────► │  FastAPI    │                   │
│  │   :80/443   │         │   :8000     │                   │
│  └─────────────┘         └──────┬──────┘                   │
│       ▲                         │                           │
│       │                         │                           │
│       │                         ▼                           │
│  ┌────┴────────────────┐  ┌────────────────────┐           │
│  │  Internet           │  │  get_embedding_    │           │
│  │  (Port 80/443)      │  │  function.py       │           │
│  └─────────────────────┘  └────────┬───────────┘           │
│                                    │                        │
│                                    │ HTTP POST              │
│                                    ▼                        │
│                           ┌─────────────────┐               │
│                           │  Ollama Server  │               │
│                           │  127.0.0.1:11434│               │
│                           │                 │               │
│                           │  /api/embed     │               │
│                           │  /api/generate  │               │
│                           └─────────────────┘               │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  File System                                 │          │
│  │                                              │          │
│  │  /opt/aflac-assist/                          │          │
│  │  ├── data/                (PDF files)        │          │
│  │  ├── chroma/              (Vector DB)        │          │
│  │  ├── app.py                                  │          │
│  │  ├── populate_database.py                    │          │
│  │  ├── query_data.py                           │          │
│  │  └── get_embedding_function.py               │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### **6. Network Security Diagram**

```
                    INTERNET (0.0.0.0/0)
                           │
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        │ Port 80          │ Port 443         │ Port 22
        │ (HTTP)           │ (HTTPS)          │ (SSH - Your IP only)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Security Group: aflac-assist-sg     │
        │                                      │
        │  Inbound Rules:                      │
        │  ┌────────────────────────────────┐  │
        │  │ 22  │ TCP │ YOUR_IP/32         │  │
        │  │ 80  │ TCP │ 0.0.0.0/0          │  │
        │  │ 443 │ TCP │ 0.0.0.0/0          │  │
        │  │ 8000│ TCP │ 0.0.0.0/0          │  │
        │  └────────────────────────────────┘  │
        │                                      │
        │  Outbound Rules:                     │
        │  ┌────────────────────────────────┐  │
        │  │ All │ All │ 0.0.0.0/0          │  │
        │  └────────────────────────────────┘  │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  EC2 Instance                        │
        │  Private IP: 10.0.1.100              │
        │  Public IP: 54.123.45.67             │
        │                                      │
        │  Listening Ports:                    │
        │  • 80 (Nginx)                        │
        │  • 443 (Nginx - SSL)                 │
        │  • 8000 (FastAPI)                    │
        │  • 11434 (Ollama - localhost only)   │
        └──────────────────────────────────────┘
```

---

### **7. Cost Breakdown Diagram**

```
┌─────────────────────────────────────────────────────────┐
│  Monthly AWS Costs (us-east-1)                          │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  EC2 Instance (t3.xlarge)                      │    │
│  │  • 4 vCPU, 16 GB RAM                           │    │
│  │  • 730 hours/month                             │    │
│  │  • $0.1664/hour                                │    │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │    │
│  │  Total: $121.47/month                          │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  EBS Storage (gp3)                             │    │
│  │  • 100 GB                                      │    │
│  │  • $0.08/GB/month                              │    │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │    │
│  │  Total: $8.00/month                            │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Data Transfer                                 │    │
│  │  • First 100 GB free                           │    │
│  │  • $0.09/GB after                              │    │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │    │
│  │  Total: ~$5/month (estimate)                   │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Elastic IP                                    │    │
│  │  • Free if attached to running instance        │    │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │    │
│  │  Total: $0/month                               │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  TOTAL ESTIMATED MONTHLY COST                  │    │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │    │
│  │  $134.47/month                                 │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  Cost Optimization Tips:                                │
│  • Use Reserved Instance: Save 30-40%                   │
│  • Use Spot Instance: Save 70-80% (with interruptions)  │
│  • Stop instance when not in use                        │
│  • Use t3 instances with burstable CPU                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Quick Reference

### **Minimum AWS Resources Needed**
1. ✅ EC2 Instance (t3.xlarge)
2. ✅ EBS Volume (100 GB)
3. ✅ Security Group
4. ✅ SSH Key Pair
5. ⭐ Elastic IP (optional but recommended)

### **Optional Resources**
- Route 53 (custom domain)
- Certificate Manager (SSL)
- S3 Bucket (backups)
- CloudWatch (monitoring)
- Application Load Balancer (HA setup)

---

## 🎯 Deployment Decision Tree

```
Do you need high availability?
│
├─ YES → Use production architecture
│         • Multiple EC2 instances
│         • Load balancer
│         • EFS for shared storage
│         • Cost: ~$500/month
│
└─ NO  → Use single EC2 instance
          │
          ├─ Need GPU for faster inference?
          │  │
          │  ├─ YES → g4dn.xlarge ($380/month)
          │  └─ NO  → t3.xlarge ($134/month)
          │
          └─ Budget tight?
             │
             ├─ YES → Use t3.large ($60/month)
             │         • Slower performance
             │         • Good for testing
             │
             └─ NO  → Use t3.xlarge ($134/month)
                       • Recommended balance
```
