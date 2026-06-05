from typing import TypedDict, Annotated
import operator
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

@tool
def calculate(expression: str) -> str:
    """Evaluate a basic math expression like 2+2 or 10*5."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

@tool
def web_search(query: str) -> str:
    """Search the web for any information including news, current events, people, and facts."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(r['body'])
        return "\n".join(results) if results else "No results found"
    except Exception as e:
        return f"Search failed: {str(e)}"

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [calculate, web_search, wiki_tool]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    
)
llm_with_tools = llm.bind_tools(tools)

tool_map = {t.name: t for t in tools}

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        t = tool_map[tool_call["name"]]
        result = t.invoke(tool_call["args"])
        from langchain_core.messages import ToolMessage
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    return {"messages": results}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

agent_app = graph.compile()

def run_agent(query: str) -> str:
    result = agent_app.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    # সব messages এ content খোঁজো, শেষেরটা নাও
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return "No response"