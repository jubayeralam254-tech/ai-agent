"""
Streamlit Frontend for the AI Support Operations Platform.
This acts as the customer-facing support ticket submission interface.

Run: streamlit run app.py
"""

import streamlit as st
import requests
import json
from uuid import UUID

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="TechCorp AI Support",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ========== STATIC CONFIGURATION ==========
API_BASE_URL = "http://127.0.0.1:8001"
API_QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
TECHCORP_ORG_ID = "11111111-1111-1111-1111-111111111111"

# ========== CUSTOM CSS ==========
st.markdown(
    """
    <style>
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        font-weight: bold;
        margin: 10px 0;
    }
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        font-weight: bold;
        margin: 10px 0;
    }
    .agent-step {
        background-color: #f0f0f0;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== PAGE HEADER ==========
st.title("🤖 TechCorp AI Support Portal")
st.markdown(
    "Submit your support ticket and let our intelligent AI agent assist you instantly."
)
st.divider()

# ========== SIDEBAR INFO ==========
with st.sidebar:
    st.header("ℹ️ About This System")
    st.markdown(
        """
    **Multi-Agent AI Support System** powered by:
    - 🧠 Google Gemini (LLM)
    - 📊 Vector Database (pgvector)
    - 🔄 LangGraph (Agent Orchestration)
    - ⚡ FastAPI (Backend)
    
    **How It Works:**
    1. You submit a ticket
    2. AI retrieves relevant knowledge base docs
    3. LangGraph decides: Auto-resolve or escalate to human?
    4. Response is delivered instantly
    
    **Portfolio Features:**
    ✅ Multi-tenant isolation
    ✅ Structured LLM outputs
    ✅ Vector similarity search
    ✅ Graceful human escalation
 
    """
    )
    st.divider()
    st.markdown("**Organization ID (TechCorp SaaS):**")
    st.code(TECHCORP_ORG_ID, language="text")

# ========== FORM SECTION ==========
st.header("📝 Submit Your Support Ticket")

# Organization ID input (pre-filled)
col1, col2 = st.columns([3, 1])
with col1:
    org_id = st.text_input(
        "Organization ID",
        value=TECHCORP_ORG_ID,
        disabled=True,
        help="Your organization's unique identifier (read-only for this demo)",
    )
with col2:
    st.caption("(Pre-filled)")

# Customer query textarea
query = st.text_area(
    "Your Support Question or Issue",
    placeholder="e.g., 'I forgot my password and need to reset it' or 'What is your refund policy?'",
    height=120,
    help="Describe your support issue in detail. The AI will search our knowledge base to find the best resolution.",
)

# Submit button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("🚀 Submit Ticket", type="primary", use_container_width=True)

st.divider()

# ========== RESPONSE HANDLING ==========
if submit_button:
    if not query.strip():
        st.error("❌ Please enter your support question before submitting.")
    else:
        # Show loading state
        with st.spinner("🔄 Processing your ticket... Please wait"):
            try:
                # Prepare request payload
                payload = {
                    "query": query,
                    "organization_id": org_id,
                    "user_id": "00000000-0000-0000-0000-000000000000",  # Demo user ID
                }

                # Send POST request to FastAPI backend
                response = requests.post(
                    API_QUERY_ENDPOINT,
                    json=payload,
                    timeout=30,
                )

                # Handle response
                if response.status_code == 200:
                    data = response.json()

                    # Extract response fields
                    ai_response = data.get("result", "No response generated.")
                    needs_human = data.get("needs_human", False)
                    agent_steps = data.get("agent_steps", [])

                    # ===== DISPLAY RESULT =====
                    st.success("✅ Ticket processed successfully!")

                    # Display status badge
                    if needs_human:
                        st.markdown(
                            '<div class="warning-badge">⚠️ Routed to Human Agent</div>',
                            unsafe_allow_html=True,
                        )
                        st.warning(
                            "This ticket requires human expertise. A support agent will review your case shortly."
                        )
                    else:
                        st.markdown(
                            '<div class="success-badge">✅ Auto-Resolved by AI</div>',
                            unsafe_allow_html=True,
                        )

                    # Display AI response
                    st.subheader("📋 Resolution")
                    st.info(ai_response)

                    # Display agent steps (expandable)
                    with st.expander("🔍 See Agent Processing Steps (Debug Info)"):
                        if agent_steps:
                            for idx, step in enumerate(agent_steps, 1):
                                st.markdown(
                                    f'<div class="agent-step">Step {idx}: {step}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No agent steps recorded.")

                    # Additional info
                    st.divider()
                    st.caption(
                        "💡 **Tip:** For faster responses, provide clear and specific details about your issue."
                    )

                else:
                    st.error(
                        f"❌ Backend error (Status {response.status_code}): {response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to backend. Is the FastAPI server running at http://127.0.0.1:8001?"
                )
                st.info(
                    "Run the backend with: `uvicorn main:app --port 8001`"
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. The backend took too long to respond.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# ========== FOOTER ==========
st.divider()
st.caption(
    "🏢 AI Support Operations Platform v1.0 | "
    "Powered by Gemini + pgvector + LangGraph | "
    "Built for portfolio demonstration"
)
