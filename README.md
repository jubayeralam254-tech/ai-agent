# AI Support Operations Platform

> **Enterprise-grade multi-tenant AI support system** — auto-resolves customer tickets using RAG, escalates complex cases to human agents, and streams responses in real time.

**Live API:** https://ai-agent-k4v6.onrender.com  
**Interactive Docs:** https://ai-agent-k4v6.onrender.com/docs  
**Frontend:** Streamlit (run locally, connects to deployed API)

---

## What This Solves

Most support systems are either fully manual (slow, expensive) or fully automated (frustrating when the bot can't help). This platform does both intelligently:

- **Auto-resolves** tickets the AI can handle confidently — instant response, no human needed
- **Escalates** tickets outside the knowledge base — flags them with a summary for human review
- **Multi-tenant** — one deployment serves multiple organizations, each with isolated knowledge bases and users

---

## Live Demo

```bash
# 1. Register
curl -X POST https://ai-agent-k4v6.onrender.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@yourcompany.com", "password": "yourpassword", 
       "organization_id": "11111111-1111-1111-1111-111111111111", "role": "customer"}'

# 2. Login → get JWT token
curl -X POST https://ai-agent-k4v6.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@yourcompany.com", "password": "yourpassword"}'

# 3. Query (standard)
curl -X POST https://ai-agent-k4v6.onrender.com/api/v1/query \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your refund policy?"}'

# 4. Query (streaming — SSE)
curl -X POST https://ai-agent-k4v6.onrender.com/api/v1/query/stream \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}' --no-buffer
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph |
| LLM | Google Gemini 2.5 Flash Lite |
| Embeddings | Gemini Embedding 001 (3072-dim) |
| Vector Search | pgvector (`<->` cosine distance) |
| Database | PostgreSQL (Neon cloud) |
| Async ORM | SQLAlchemy (asyncpg driver) |
| Authentication | JWT (python-jose + bcrypt) |
| Rate Limiting | SlowAPI (per-IP) |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |

---

## Architecture
Customer Query

│

▼

FastAPI  ──  JWT Auth Middleware

│

▼

LangGraph Agent

┌─────────────────────────────────────┐

│  Node 1: retrieve_context           │  ← Embeds query → pgvector similarity search

│  Node 2: ai_responder               │  ← Gemini LLM + structured Pydantic output

└─────────────────────────────────────┘

│

├── can_resolve = True  →  Stream answer to customer + log ticket

└── can_resolve = False →  Flag for human agent + log escalated ticket

**Multi-tenancy:** Every query is scoped by `organization_id` extracted from the JWT token. The vector search enforces `WHERE d.organization_id = :org_id` — one organization can never access another's knowledge base.

**Structured LLM Output:** The LLM returns a Pydantic-validated object, not free text:
```python
class ResolutionDecision(BaseModel):
    can_resolve: bool
    response_draft: str
    confidence_score: float  # 0.0 to 1.0
```

---

## API Reference

### Authentication

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | Public | Register new user |
| POST | `/auth/login` | Public | Login → JWT token |

### Core

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/query` | JWT | Submit support ticket |
| POST | `/api/v1/query/stream` | JWT | Submit ticket (SSE streaming) |
| GET | `/api/v1/tickets` | JWT | List org's tickets |
| GET | `/api/v1/documents` | JWT | List knowledge base docs |
| POST | `/api/v1/documents` | Admin/Agent | Upload new KB document |

### Admin

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/admin/organizations` | Admin | Create new tenant |
| GET | `/api/v1/admin/organizations` | Admin | List all tenants |
| GET | `/api/v1/admin/tickets/escalated` | Admin | View escalated tickets |

### System

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | Public | DB + service health check |

---

### Example Responses

**Auto-resolved ticket:**
```json
{
  "result": "To reset your password, visit /reset-password on the login page...",
  "needs_human": false,
  "agent_steps": [
    "Started Support Workflow",
    "Retrieved context from Vector DB",
    "AI Decision: Resolve=True, Confidence=1.0"
  ],
  "ticket_id": "506b1da5-4fbc-41d8-8ed8-3421af938272"
}
```

**Escalated ticket:**
```json
{
  "result": "Customer is asking about enterprise billing. Requires account manager review.",
  "needs_human": true,
  "agent_steps": ["..."],
  "ticket_id": "758e4618-f63f-43c4-ab98-e68a41921801"
}
```

**Streaming response (SSE):**
data: {"type": "step", "content": "Started Support Workflow"}

data: {"type": "step", "content": "Retrieving knowledge base context..."}

data: {"type": "step", "content": "Streaming response..."}

data: {"type": "token", "content": "Our "}

data: {"type": "token", "content": "refund "}

data: {"type": "token", "content": "policy "}

...

data: {"type": "done", "needs_human": false, "ticket_id": "3c73f6f6-..."}

---

## Project Structure
├── main.py           # FastAPI app — all routes, middleware, rate limiting

├── agent.py          # LangGraph workflow (retrieve → respond → END)

├── auth.py           # JWT token creation, password hashing, auth dependency

├── database.py       # Async SQLAlchemy engine + session factory

├── models.py         # ORM models: Organization, User, Document, DocumentChunk, Ticket

├── schemas.py        # Pydantic request/response schemas

├── app.py            # Streamlit frontend (login, register, query UI)

├── seed_fixed.py     # Seeds DB with sample org + 3 knowledge base documents

├── reset_db.py       # Safe schema migration (adds tables/columns, no data loss)

├── Dockerfile        # Production container

├── docker-compose.yml # Local dev: API + PostgreSQL with pgvector

├── render.yaml       # Render deployment config

└── requirements.txt

---

## Local Setup

### Option A: Docker (Recommended)

```bash
git clone https://github.com/jubayeralam254-tech/ai-agent.git
cd ai-agent

# Create .env
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Start everything
docker compose up --build
```

API runs at `http://localhost:8001`  
Docs at `http://localhost:8001/docs`

### Option B: Manual

**Prerequisites:** Python 3.10+, PostgreSQL with pgvector, Gemini API key

```bash
git clone https://github.com/jubayeralam254-tech/ai-agent.git
cd ai-agent
pip install -r requirements.txt
```

Create `.env`:
```env
GEMINI_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/ai_support_db
JWT_SECRET_KEY=your-secret-key-here
```

```bash
# Create DB
psql -U postgres -c "CREATE DATABASE ai_support_db;"

# Start API (auto-creates tables on startup)
uvicorn main:app --port 8001 --reload

# Seed knowledge base
python seed_fixed.py

# Run frontend
streamlit run app.py
```

---

## Deployment

Deployed on **Render** via `render.yaml`. Auto-deploys on every push to `main`.

### Environment Variables (Render Dashboard)

| Key | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `DATABASE_URL` | `postgresql+asyncpg://USER:PASS@HOST/DB?ssl=require` |
| `JWT_SECRET_KEY` | Strong random string for JWT signing |

---

## Database Schema
organizations

└── id (UUID PK), name
users  (FK → organizations)

└── id, organization_id, email, role, hashed_password
documents  (FK → organizations)

└── id, organization_id, title, source_url
document_chunks  (FK → documents)

└── id, document_id, content, chunk_index

└── embedding  Vector(3072)  ← pgvector
tickets  (FK → organizations, users)

└── id, organization_id, user_id

└── query, response, needs_human, created_at

---

## Key Technical Decisions

**Why raw SQL for vector search?**  
asyncpg requires explicit pgvector type registration with SQLAlchemy ORM. `CAST(:embedding AS vector)` in raw SQL bypasses this cleanly and reliably.

**Why `run_in_executor` for embeddings?**  
`genai.embed_content()` is synchronous. Wrapping it prevents blocking the async event loop under concurrent requests.

**Why SSE over WebSockets for streaming?**  
SSE is unidirectional and HTTP-native — simpler to implement, works through proxies, and sufficient for one-way token streaming. WebSockets would add complexity without benefit here.

**Why LangGraph over vanilla LLM calls?**  
The graph structure makes the retrieve → respond pipeline explicit, testable, and extensible. Adding new nodes (e.g. conversation history, tool calling) requires minimal changes to the existing graph.

---

## Roadmap

- [ ] Conversation history (multi-turn context)
- [ ] Tool-calling agents (calendar, CRM integrations)
- [ ] Webhook notifications for escalated tickets
- [ ] Analytics dashboard (resolution rate, avg confidence)

---

**Built by:** Jubayer Alam  
**Stack:** FastAPI · LangGraph · pgvector · Gemini API · JWT · SSE · Docker · Streamlit


