import os
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from requests.exceptions import HTTPError, Timeout, ConnectionError
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import uuid

import logging
import time

load_dotenv()
api_key = os.getenv("NEWSDATA_API_KEY")

if not api_key:
    raise ValueError("NEWSDATA_API_KEY not found")

class RequestNews(BaseModel):
    category: str

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@tool
def get_news(category: str) -> str:
    """Get the latest news articles for a given category (e.g. business, science, technology, sports)."""
    logger.info(f"get_news called with category={category}")
    try:
        start = time.time()
        response = requests.get(
        f"https://newsdata.io/api/1/latest?apikey={api_key}&category={category}&language=en", timeout=10
        )
        response.raise_for_status()
        logger.info(f"get_news completed in {time.time() - start:.2f}s")
    except Timeout:
        logger.error(f"get_news timed out for category={category}")
        return f"News Data service timed out for {category}. Try again later."
    except HTTPError as e:
        logger.error(f"get_news http error for category={category}: {e}")
        return f"News Data service returned error: {e}"
    except ConnectionError as e:
        logger.error(f"get_news connection error for category={category}: {e}")
        return f"News Data service connection error: {e}"
    data = response.json()

    articles = data.get("results", [])

    selected = [{
            "id": a.get("article_id"),
            "title": a.get("title"),
            "link": a.get("link"),
            "description": a.get("description"),
            "image": a.get("image_url"),
            "country": a.get("country"),
            "category": a.get("category"),
            "pubDate": a.get("pubDate")
        } for a in articles]

    formatted = ""
    for a in selected:
        formatted += f"- {a['title']}\n  {a['link']}\n  {a['pubDate']}\n\n"
    return formatted.strip()

tools = [get_news]
tool_node = ToolNode(tools)
llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent_node(state: MessagesState) -> MessagesState:
    """"""
    logger.info("agent_node called")
    system = SystemMessage(
        content="You are helpful assistant. Use tools when needed."
    )
    response = llm_with_tools.invoke([system] + state["messages"])
    logger.info(f"LLM response: {response.content[:100]}")
    return {"messages": [response]}


graph = StateGraph(MessagesState)

graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "agent_node")
graph.add_edge("tool_node", "agent_node")
graph.add_conditional_edges("agent_node", tools_condition, {
    "tools": "tool_node",
    "__end__": END
})

checkpointer = MemorySaver()
app_graph = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tool_node"]
)

app = FastAPI()

@app.post("/news")
def request_news(category: RequestNews):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    app_graph.invoke(
        {"messages": [HumanMessage(content=f"What is the latest {category.category} news?")]},
        config=config
    )
    return {"thread_id": thread_id, "status": "pending approval"}

@app.post("/news/approve")
def approval(approved: bool, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    if approved:
        results = app_graph.invoke(None, config=config)
        return results["messages"][-1].content
    return {"status": "rejected"}
