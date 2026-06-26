"""
Streamlit Frontend — AI Support Operations Platform v2.0
"""
import streamlit as st
import requests

st.set_page_config(
    page_title="TechCorp AI Support",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "https://ai-agent-k4v6.onrender.com"
TECHCORP_ORG_ID = "11111111-1111-1111-1111-111111111111"

if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

st.markdown("""
<style>
.success-badge { background-color:#d4edda; color:#155724; padding:10px 15px;
    border-radius:5px; border-left:4px solid #28a745; font-weight:bold; margin:10px 0; }
.warning-badge { background-color:#fff3cd; color:#856404; padding:10px 15px;
    border-radius:5px; border-left:4px solid #ffc107; font-weight:bold; margin:10px 0; }
.agent-step { background-color:#f0f0f0; padding:8px 12px; border-radius:4px;
    margin:5px 0; font-family:monospace; font-size:12px; }
</style>
""", unsafe_allow_html=True)


def do_register(email, password, org_id):
    try:
        resp = requests.post(f"{API_BASE_URL}/auth/register", json={
            "email": email, "password": password,
            "organization_id": org_id, "role": "customer"
        }, timeout=15)
        if resp.status_code == 201:
            st.success("✅ Registered! Now log in.")
        else:
            st.error(f"Registration failed: {resp.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def do_login(email, password):
    try:
        resp = requests.post(f"{API_BASE_URL}/auth/login", json={
            "email": email, "password": password
        }, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.token = data["access_token"]
            st.session_state.user_email = email
            return True
        else:
            st.error(f"Login failed: {resp.json().get('detail', 'Invalid credentials')}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
    return False


# ── SIDEBAR ──
with st.sidebar:
    st.header("ℹ️ About")
    if st.session_state.token:
        st.success(f"✅ Logged in:\n{st.session_state.user_email}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.user_email = None
            st.rerun()
        st.divider()
    st.markdown("""
**Multi-Tenant AI Support** powered by:
- 🧠 Google Gemini (LLM)
- 📊 pgvector (Vector DB)
- 🔄 LangGraph (Agent)
- ⚡ FastAPI (Backend)
- 🔐 JWT Authentication

✅ Multi-tenant isolation  
✅ JWT Authentication  
✅ Structured LLM outputs  
✅ Ticket logging  
✅ Human escalation
""")
    st.divider()
    st.markdown("**TechCorp Org ID:**")
    st.code(TECHCORP_ORG_ID)


# ── MAIN ──
st.title("🤖 TechCorp AI Support Portal")
st.markdown("Enterprise AI support with intelligent auto-resolution and human escalation.")
st.divider()

if not st.session_state.token:
    st.header("🔐 Login or Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email", key="l_email")
        password = st.text_input("Password", type="password", key="l_pass")
        if st.button("Login", type="primary", use_container_width=True):
            if email and password:
                if do_login(email, password):
                    st.rerun()
            else:
                st.warning("Enter email and password.")

    with tab2:
        r_email = st.text_input("Email", key="r_email")
        r_pass = st.text_input("Password (min 8 chars)", type="password", key="r_pass")
        r_org = st.text_input("Organization ID", value=TECHCORP_ORG_ID, key="r_org")
        if st.button("Register", type="primary", use_container_width=True):
            if r_email and r_pass and r_org:
                do_register(r_email, r_pass, r_org)
            else:
                st.warning("Fill all fields.")

else:
    st.header("📝 Submit Support Ticket")
    query = st.text_area(
        "Your Question",
        placeholder="e.g., How do I reset my password?",
        height=120,
    )
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        submit = st.button("🚀 Submit", type="primary", use_container_width=True)
    st.divider()

    if submit:
        if not query.strip():
            st.error("❌ Enter your question first.")
        else:
            with st.spinner("🔄 Processing..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/api/v1/query",
                        json={"query": query},
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("✅ Ticket processed!")
                        if data.get("needs_human"):
                            st.markdown('<div class="warning-badge">⚠️ Routed to Human Agent</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-badge">✅ Auto-Resolved by AI</div>', unsafe_allow_html=True)
                        st.subheader("📋 Resolution")
                        st.info(data.get("result"))
                        if data.get("ticket_id"):
                            st.caption(f"Ticket ID: `{data['ticket_id']}`")
                        with st.expander("🔍 Agent Steps"):
                            for i, step in enumerate(data.get("agent_steps", []), 1):
                                st.markdown(f'<div class="agent-step">Step {i}: {step}</div>', unsafe_allow_html=True)
                    elif resp.status_code == 401:
                        st.error("❌ Session expired. Please log in again.")
                        st.session_state.token = None
                        st.rerun()
                    else:
                        st.error(f"❌ Error {resp.status_code}: {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout. Try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

st.divider()
st.caption("🏢 AI Support Platform v2.0 | Gemini + pgvector + LangGraph + JWT")