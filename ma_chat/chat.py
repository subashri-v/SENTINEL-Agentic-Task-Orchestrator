import uuid
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from . import config
from . import database
from .graph import workflow

# =====================================================
# SYSTEM CHAT RUNNER LOOP
# =====================================================
async def chat(app):
    print("\n==================================================")
    print("End-User Channel Active (Non-Blocking Async).")
    print("==================================================")

    while True:
        # Wrap synchronous prompt input to avoid starving the event loop
        user_input = await asyncio.to_thread(input, "\n[User] Ask a question (or 'quit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}

        payload = {
            "question": user_input,
            "messages": [HumanMessage(content=user_input)],
            "retry_count": 0
        }

        print(f"Processing Request under Thread ID: {thread_id} ...")
        result = await app.ainvoke(payload, config=thread_config)

        snapshot = await app.aget_state(thread_config)
        if snapshot.next:
            print(f"⚠️ [Status] Thread {thread_id} requires supervisor approval.")
            print("Your query has been queued for evaluation. You may continue typing new questions.")
            continue

        print(f"\n[Answer]: {result.get('answer')}")
        await database.log_interaction_to_hotl(result)

        print("\n" + "="*20 + " CITATIONS " + "="*20)
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                print("-", c)
        else:
            print("No background documents or endpoints referenced.")

async def main():
    # 1. Initialize PostgreSQL schemas & background MCP configurations
    await database.init_db_and_mcp()

    # 2. Extract and keep the checkpointer context persistent
    cp_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
    checkpointer = await cp_context.__aenter__()

    # 3. Compile the graph onto the shared config.app reference
    config.app = workflow.compile(checkpointer=checkpointer)

    print("\n[System] LangGraph Agent Pipeline Compiled Successfully.")

    try:
        await chat(config.app)
    finally:
        # 4. Clean up all persistent pools gracefully on hard application exits
        await cp_context.__aexit__(None, None, None)

        if config.conn:
            print("[System] Closing PostgreSQL primary connections...")
            await config.conn.close()
