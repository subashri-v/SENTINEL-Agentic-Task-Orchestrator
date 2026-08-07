import asyncio
import selectors

# async_cli.py, async_rev.py, and test_env.py import these names directly
# from this module — keep them re-exported here after the ma_chat/ split.
from ma_chat.config import DB_CONFIG
from ma_chat.state import AgentState
from ma_chat.database import log_interaction_to_hotl
from ma_chat.graph import workflow, get_app, build_app
from ma_chat.chat import chat, main

if __name__ == "__main__":
    # Configure asyncio to use SelectorEventLoop to satisfy psycopg async connections
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
    )
