
import os
import operator
import asyncio
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from uuid import UUID
from functools import lru_cache
import hashlib

# Simple in-memory cache (production-এ Redis ব্যবহার করো)
_embedding_cache: dict = {}

async def embed_with_cache_and_retry(query_text: str) -> dict:
    cache_key = hashlib.md5(query_text.encode()).hexdigest()
    
    if cache_key in _embedding_cache:
        print(f"DEBUG: Cache hit for query")
        return _embedding_cache[cache_key]
    
    result = await embed_with_retry(query_text)
    _embedding_cache[cache_key] = result
    return result
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from sqlalchemy import text
from pydantic import BaseModel, Field

from database import AsyncSessionLocal


class ResolutionDecision(BaseModel):
    can_resolve: bool = Field(description="True if the AI can confidently resolve the ticket.")
    response_draft: str = Field(description="The drafted response or summary for human agent.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")


class AgentState(TypedDict):
    organization_id: UUID
    query: str
    retrieved_context: str
    steps_taken: Annotated[List[str], operator.add]
    final_response: str
    needs_human: bool


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

structured_llm = llm.with_structured_output(ResolutionDecision)


# ─── RETRY HELPERS ────────────────────────────────────────────────────────────

async def embed_with_retry(query_text: str, max_retries: int = 3) -> dict:
    """
    Calls Gemini embed_content with exponential backoff on 429 rate-limit errors.
    Waits 2^attempt seconds between retries (1s → 2s → 4s).
    """
    loop = asyncio.get_running_loop()
    for attempt in range(max_retries):
        try:
            embed_result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=query_text
                )
            )
            return embed_result
        except Exception as e:
            is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = 2 ** attempt   # 1s, 2s, 4s
                print(f"DEBUG: 429 on embedding (attempt {attempt + 1}), retrying in {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)
            else:
                raise


async def llm_invoke_with_retry(messages: list, max_retries: int = 3) -> ResolutionDecision:
    """
    Calls structured_llm.ainvoke with exponential backoff on 429 rate-limit errors.
    """
    for attempt in range(max_retries):
        try:
            return await structured_llm.ainvoke(messages)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = 2 ** attempt
                print(f"DEBUG: 429 on LLM call (attempt {attempt + 1}), retrying in {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)
            else:
                raise


# ─── NODES ────────────────────────────────────────────────────────────────────

async def retrieve_context_node(state: AgentState) -> dict:
    query_text = state["query"]
    org_id = state["organization_id"]

    try:
        embed_result = await embed_with_cache_and_retry(query_text)
        query_embedding = embed_result['embedding']
        print(f"DEBUG: embedding len={len(query_embedding)}")
    except Exception as e:
        print(f"DEBUG: embedding error={str(e)}")
        return {"retrieved_context": f"Error: {str(e)}", "steps_taken": ["Embedder Failure"]}

    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    retrieved_docs = []
    async with AsyncSessionLocal() as session:
        stmt = text("""
            SELECT dc.content
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.organization_id = :org_id
            ORDER BY dc.embedding <-> CAST(:embedding AS vector)
            LIMIT 3
        """)
        result = await session.execute(stmt, {
            "org_id": str(org_id),
            "embedding": embedding_str
        })
        rows = result.fetchall()
        retrieved_docs = [row[0] for row in rows]
        print(f"DEBUG: org_id={org_id}, retrieved {len(retrieved_docs)} chunks")

    context_str = "\n---\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."

    return {
        "retrieved_context": context_str,
        "steps_taken": ["Retrieved context from Vector DB"]
    }


async def ai_responder_node(state: AgentState) -> dict:
    system_prompt = (
        "You are an expert Enterprise Support AI.\n"
        "Analyze the user's query against the provided Knowledge Base context.\n"
        "If you can confidently solve the problem based on the context, set 'can_resolve' to True and provide a polite, accurate 'response_draft'.\n"
        "If the query is too complex or isn't answered in the context, set 'can_resolve' to False and provide a short summary for the human agent."
    )

    human_prompt = f"Query: {state['query']}\n\nContext:\n{state['retrieved_context']}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    try:
        decision: ResolutionDecision = await llm_invoke_with_retry(messages)
    except Exception as e:
        print(f"DEBUG: LLM error after retries={str(e)}")
        return {
            "final_response": "Service temporarily unavailable. Please try again in a moment.",
            "needs_human": True,
            "steps_taken": ["LLM Failure — routed to human agent"]
        }

    needs_human = not decision.can_resolve

    return {
        "final_response": decision.response_draft,
        "needs_human": needs_human,
        "steps_taken": [f"AI Decision: Resolve={decision.can_resolve}, Confidence={decision.confidence_score}"]
    }


# ─── GRAPH ────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("rag_router", retrieve_context_node)
    workflow.add_node("ai_responder", ai_responder_node)
    workflow.set_entry_point("rag_router")
    workflow.add_edge("rag_router", "ai_responder")
    workflow.add_edge("ai_responder", END)
    return workflow.compile()


agent_app = build_graph()


async def run_support_agent(query: str, organization_id: UUID) -> dict:
    initial_state = {
        "organization_id": organization_id,
        "query": query,
        "retrieved_context": "",
        "steps_taken": ["Started Support Workflow"],
        "final_response": "",
        "needs_human": False
    }

    final_state = await agent_app.ainvoke(initial_state)

    return {
        "response": final_state["final_response"],
        "needs_human": final_state["needs_human"],
        "steps": final_state["steps_taken"]
    }
