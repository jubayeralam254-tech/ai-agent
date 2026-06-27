# AI Support Operations Platform

Multi-tenant AI support API — auto-resolves customer tickets 
using RAG, escalates complex cases to human agents.

**Live:** https://ai-agent-k4v6.onrender.com/docs

## Stack
FastAPI · LangGraph · pgvector · Gemini API · JWT · Docker

## Quick Test
# Register
curl -X POST https://ai-agent-k4v6.onrender.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@yourcompany.com", "password": "password123",
       "organization_id": "11111111-1111-1111-1111-111111111111", "role": "customer"}'

# Login → copy the access_token
curl -X POST https://ai-agent-k4v6.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@yourcompany.com", "password": "password123"}'

# Query
curl -X POST https://ai-agent-k4v6.onrender.com/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_token>" \
  -d '{"query": "What is your refund policy?"}'

## What It Does
- JWT authentication (register/login)
- RAG pipeline with pgvector similarity search  
- Auto-resolve or escalate with confidence score
- SSE streaming responses
- Document upload API
- Admin endpoints for tenant management
