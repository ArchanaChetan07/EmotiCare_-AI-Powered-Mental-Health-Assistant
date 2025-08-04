from langchain.tools import tool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

from .data_loader import load_gold_data

# 🧠 Tool 1: Query GoEmotions dataset
@tool
def search_goemotions(query: str) -> str:
    """
    Search for a keyword in the goemotions_gold dataset.
    Returns top 3 results with emotion scores.
    """
    data = load_gold_data()
    df = data["goemotions"]

    results = df[df["text"].str.contains(query, case=False, na=False)]
    if results.empty:
        return "❌ No matching emotions found."

    emotion_cols = [col for col in df.columns if col not in ["text", "id"]]
    return results[["text"] + emotion_cols].head(3).to_string(index=False)

# 🛠️ Register tools
def get_tools():
    """
    Return the list of tools to be used in the chatbot:
    - search_goemotions: local dataset
    - TavilySearchResults: web search
    """
    return [
        Tool.from_function(search_goemotions),
        TavilySearchResults(max_results=2)
    ]

# 🔁 Wrap them in a LangGraph ToolNode
def create_tool_node(tools):
    return ToolNode(tools=tools)
