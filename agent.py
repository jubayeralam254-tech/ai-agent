import json
import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List, Optional
from uuid import UUID

# Load environment variables
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Local imports
from database import AsyncSessionLocal
from models import DocumentChunk, Document
from sqlalchemy import select

# --- SCHEMAS ---

# Pydantic schema for structured output from Gemini
class ResolutionDecision(BaseModel):
    can_resolve: bool = Field(description="True if the AI can confidently resolve the ticket using the retrieved context. False if it needs human support.")
    response_draft: str = Field(description="The drafted response to the customer or a summary for the human agent.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")

# Define the unified State for LangGraph
class AgentState(TypedDict):
    organization_id: UUID
    query: str
    retrieved_context: str
    steps_taken: Annotated[List[str], list.append]
    final_response: str
    needs_human: bool

# --- AI SETUP ---

# Use an async-compatible Gemini model setup
# Gemini 1.5/2.0 Flash is extremely fast and capable of structured output
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

# Use Google's embedding model to match our database (e.g., text-embedding-004)
# Ensure the model outputs 768 dimensions as configured in models.py
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Bind the Pydantic schema directly to the model to force structured JSON output
structured_llm = llm.with_structured_output(ResolutionDecision)

# --- GRAPH NODES ---

async def retrieve_context_node(state: AgentState) -> dict:
    """
    RAG Step: Embed the query, query Postges (pgvector) using async SQLAlchemy,
    and return the context filtered by the user's organization.
    """
    query_text = state["query"]
    org_id = state["organization_id"]
    
    # 1. Embed the incoming query via Google API
    try:
         query_embedding = await embeddings.aembed_query(query_text)
    except Exception as e:
         return {"retrieved_context": f"Error embedding query: {str(e)}", "steps_taken": ["Embedder Failure"]}

    # 2. Query Postgres for closest embeddings using L2 distance (<->)
    # We strictly filter by organization_id for tenant isolation
    retrieved_docs = []
    async with AsyncSessionLocal() as session:
        # Construct raw pgvector distance query
        stmt = (
            select(DocumentChunk.content)
            .join(DocumentChunk.document)
            .where(Document.organization_id == org_id)
            .where(DocumentChunk.embedding.l2_distance(query_embedding) < 0.7) # Distance threshold
            .order_by(DocumentChunk.embedding.l2_distance(query_embedding))
            .limit(3)
        )
        
        result = await session.execute(stmt)
        chunks = result.scalars().all()
        retrieved_docs = list(chunks)

    context_str = "\n---\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."
    
    return {
        "retrieved_context": context_str,
        "steps_taken": ["Retrieved context from Vector DB"]
    }

async def ai_responder_node(state: AgentState) -> dict:
    """
    Decision Step: Analyzes the query and context. 
    Decides whether to deflect the ticket or route to human.
    """
    system_prompt = (
        "You are an expert Enterprise Support AI.\n"
        "Analyze the user's query against the provided Knowledge Base context.\n"
        "If you can confidently solve the problem based on the context, set 'can_resolve' to True and provide a polite, accurate 'response_draft'.\n"
        "If the query is too complex, touches on sensitive legal/billing edge cases, or isn't answered in the context, set 'can_resolve' to False and provide a short summary in 'response_draft' for the human agent."
    )
    
    human_prompt = f"Query: {state['query']}\n\nContext:\n{state['retrieved_context']}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    # Invoke the Gemini model expecting the ResolutionDecision Pydantic object
    decision: ResolutionDecision = await structured_llm.ainvoke(messages)
    
    needs_human = not decision.can_resolve
    
    return {
        "final_response": decision.response_draft,
        "needs_human": needs_human,
        "steps_taken": [f"AI Decision: Resolve={decision.can_resolve}, Confidence={decision.confidence_score}"]
    }

# --- LANGGRAPH ORCHESTRATION ---

def build_graph() -> StateGraph:
    """Constructs the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("rag_router", retrieve_context_node)
    workflow.add_node("ai_responder", ai_responder_node)
    
    # Define Edges (Simplest Flow)
    workflow.set_entry_point("rag_router")
    workflow.add_edge("rag_router", "ai_responder")
    workflow.add_edge("ai_responder", END)
    
    # Compile the graph
    return workflow.compile()

agent_app = build_graph()

# --- ENTRY POINT EXPOSED TO FASTAPI ---

async def run_support_agent(query: str, organization_id: UUID) -> dict:
    """
    Async entry point for FastAPI routes.
    """
    initial_state = {
        "organization_id": organization_id,
        "query": query,
        "retrieved_context": "",
        "steps_taken": ["Started Support Workflow"],
        "final_response": "",
        "needs_human": False
    }
    
    # Execute the graph asynchronously
    final_state = await agent_app.ainvoke(initial_state)
    
    return {
        "response": final_state["final_response"],
        "needs_human": final_state["needs_human"],
        "steps": final_state["steps_taken"]
    }