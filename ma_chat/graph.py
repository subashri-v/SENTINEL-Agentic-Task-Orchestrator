from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from . import config
from . import database
from . import nodes
from .state import AgentState

# =====================================================
# BUILDING THE LANGGRAPH PIPELINE
# =====================================================
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", nodes.node_orchestrator)
workflow.add_node("RAG", nodes.node_rag_agent)
workflow.add_node("TAVILY", nodes.node_tavily_agent)
workflow.add_node("GITHUB", nodes.node_github_agent)
workflow.add_node("SUMMARY_AGENT", nodes.node_summary_agent)
workflow.add_node("evaluator", nodes.node_evaluator)
workflow.add_node("human_intervention", nodes.node_human_intervention)
workflow.add_node("MATH", nodes.node_math_agent)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    nodes.route_decision,
    {
        "RAG": "RAG",
        "TAVILY": "TAVILY",
        "GITHUB": "GITHUB",
        "SUMMARY_AGENT": "SUMMARY_AGENT",
        "MATH": "MATH"
    }
)

workflow.add_edge("RAG", "evaluator")
workflow.add_edge("TAVILY", "evaluator")
workflow.add_edge("GITHUB", "evaluator")
workflow.add_edge("SUMMARY_AGENT", "evaluator")
workflow.add_edge("MATH", "evaluator")

workflow.add_conditional_edges(
    "evaluator",
    nodes.evaluate_decision,
    {
        "ACCEPT": END,
        "RAG": "RAG",
        "GITHUB": "GITHUB",
        "TAVILY": "TAVILY",
        "SUMMARY_AGENT": "SUMMARY_AGENT",
        "INTERVENT": "human_intervention",
        "MATH": "MATH"
    }
)
workflow.add_edge("human_intervention", END)


def get_app():
    """
    Synchronous fallback or retrieval. Note: Because AsyncSqliteSaver
    requires an active async connection pool, compiling a fresh instance
    with a long-lived connection is much safer for secondary tools like the UI.
    """
    if config.app is None:
        # Fallback setup: create a standard connection pool for the checkpointer
        # so it stays alive across multiple independent UI calls.
        cp = AsyncSqliteSaver.from_conn_string("checkpoints.db")
        config.app = workflow.compile(checkpointer=cp)
    return config.app

async def build_app():
    await database.init_db()
    # Create the checkpointer without the 'async with' auto-close block
    # if you intend to reuse 'app' outside of this function scope.
    cp = AsyncSqliteSaver.from_conn_string("checkpoints.db")
    config.app = workflow.compile(checkpointer=cp)
    return config.app
