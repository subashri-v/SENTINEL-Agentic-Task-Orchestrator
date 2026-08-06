import os
import json
import asyncio
import sqlite3
from async_ma_chat import build_app  # Ensure this exposes the app built via AsyncSqliteSaver
import psycopg

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, ListView, ListItem, Label, TextArea, Button, Static
from textual.reactive import reactive

from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage

# Import your async database configs and logger explicitly
from async_ma_chat import DB_CONFIG, log_interaction_to_hotl


class ReviewerDashboard(App):
    CSS = """
    Screen {
        background: #1a1a1a;
    }
    .panel-title {
        background: #333333;
        color: #00ff00;
        text-align: center;
        text-style: bold;
        padding: 1;
    }
    #queue-pane {
        width: 30%;
        border-right: solid #333333;
    }
    #details-pane {
        width: 40%;
        border-right: solid #333333;
        padding: 1;
    }
    #history-pane {
        width: 30%;
        padding: 1;
    }
    .meta-label {
        color: #ffaa00;
        text-style: bold;
        margin-top: 1;
    }
    .content-box {
        background: #262626;
        border: solid #444444;
        min-height: 4;
        padding: 1;
        margin-bottom: 1;
    }
    #action-input {
        height: 6;
        margin-top: 1;
        border: solid #00ff00;
    }
    #submit-btn {
        background: #008800;
        color: white;
        margin-top: 1;
    }
    """

    active_threads = reactive([])
    selected_thread_id = reactive(None)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            # Pane 1: Dynamic Thread Queues
            with Vertical(id="queue-pane"):
                yield Label("🚨 HITL ESCALATION QUEUE", classes="panel-title")
                yield ListView(id="thread-list")
                yield Button("🔄 Refresh Queue", id="refresh-btn")

            # Pane 2: Workspace Desk (Reviewer interaction area)
            with Vertical(id="details-pane"):
                yield Label("🛠️ EVALUATION WORKSPACE", classes="panel-title")
                yield Label("User Question:", classes="meta-label")
                yield Static("Select a thread to view details.", id="view-question", classes="content-box")
                
                yield Label("AI Drafted Answer:", classes="meta-label")
                yield Static("N/A", id="view-draft", classes="content-box")
                
                yield Label("System Critique / Evaluation Failure Reason:", classes="meta-label")
                yield Static("N/A", id="view-critique", classes="content-box")
                
                yield Label("Provide Corrective Guidance / Direct Answer Override:", classes="meta-label")
                yield TextArea(id="action-input")
                yield Button("Dispatch Resolution", id="submit-btn", variant="success")

            # Pane 3: Immutable Session Records
            with Vertical(id="history-pane"):
                yield Label("📜 THREAD CHAT HISTORY", classes="panel-title")
                yield Static("No active thread history loaded.", id="history-box")

        yield Footer()

    async def on_unmount(self) -> None:
        """Ensures the SQLite connection drops cleanly when closing the Textual app."""
        if hasattr(self, 'cp_context'):
            await self.cp_context.__aexit__(None, None, None)

    async def on_mount(self) -> None:
        self.title = "LangGraph Human-in-the-Loop Orchestration Desk"
        
        # 1. Grab your raw, uncompiled workflow object
        from async_ma_chat import workflow
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        
        # 2. Assign the context manager object to a local variable
        self.cp_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
        
        # 3. Enter the async context to extract the real saver
        cp = await self.cp_context.__aenter__()
        
        # 4. Compile using the true saver instance assigned to self.app
        self.graph_app = workflow.compile(checkpointer=cp)
        
        # 5. Populate your queue!
        await self.refresh_queue()

    async def refresh_queue(self) -> None:
        """Asynchronously queries the checkpoints database for active evaluation halts."""
        try:
            # FIX: Pull unique threads from your active SQLite checkpoints.db file
            def _get_unique_threads():
                with sqlite3.connect("checkpoints.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
                    return [row[0] for row in cursor.fetchall()]

            threads = await asyncio.to_thread(_get_unique_threads)
            active_hitl_threads = []
                        
            for tid in threads:
                config = {"configurable": {"thread_id": tid}}
                snapshot = await self.graph_app.aget_state(config)
                if snapshot.next and "human_intervention" in snapshot.next:
                    active_hitl_threads.append(tid)

            self.active_threads = active_hitl_threads
            
            list_view = self.query_one("#thread-list", ListView)
            for item in list_view.query(ListItem):
                item.remove()

            for tid in self.active_threads:
                list_view.append(ListItem(Label(f"🧵 id: ...{tid[-12:]}"), id=f"id-{tid}"))
        except Exception as e:
            self.notify(f"Queue Sync Failed: {str(e)}", severity="error")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-btn":
            await self.refresh_queue()
            self.notify("Queue Refreshed")
        elif event.button.id == "submit-btn":
            await self.submit_resolution()

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item and event.item.id:
            self.selected_thread_id = event.item.id.replace("id-", "")
            await self.load_thread_workspace(self.selected_thread_id)

    async def load_thread_workspace(self, thread_id: str) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph_app.aget_state(config)

        question_text, draft_text, critique_text = "N/A", "N/A", "N/A"
        for task in snapshot.tasks:
            if task.interrupts:
                details = task.interrupts[0].value
                question_text = details.get("question", "N/A")
                draft_text = details.get("generated_answer", "N/A")
                critique_text = details.get("critique", "N/A")

        self.query_one("#view-question", Static).update(question_text)
        self.query_one("#view-draft", Static).update(draft_text)
        self.query_one("#view-critique", Static).update(critique_text)

        history_log = []
        messages = snapshot.values.get("messages", [])
        for msg in messages:
            sender = "👤 User" if msg.type == "human" else "🤖 Agent"
            history_log.append(f"[b]{sender}:[/b]\n{msg.content}\n")
        
        history_string = "\n".join(history_log) if history_log else "Empty lifecycle conversation state."
        self.query_one("#history-box", Static).update(history_string)

    async def submit_resolution(self) -> None:
        if not self.selected_thread_id:
            self.notify("No active thread selection highlighted.", severity="warning")
            return

        text_area = self.query_one("#action-input", TextArea)
        resolution_input = text_area.text.strip()

        if not resolution_input:
            self.notify("Resolution description input cannot remain empty.", severity="error")
            return

        try:
            config = {"configurable": {"thread_id": self.selected_thread_id}}
            
            # FIX: Changed 'app.ainvoke' to 'self.app.ainvoke'
            final_result = await self.graph_app.ainvoke(Command(resume=resolution_input), config=config)
            
            self.notify("Resolution dispatched cleanly!", title="Thread Finalized")
            text_area.text = "" 
            
            log_payload = {
                "question": final_result.get("question") or "Override Question",
                "answer": final_result.get("answer") or resolution_input,
                "next_agent": "ACCEPT", # Kept for backward compatibility
                "final_route": "ACCEPT", # Matches your exact PG column name mapping
                "retry_count": final_result.get("retry_count", 0),
                "system_critique": final_result.get("system_critique") or "Human Intervention Override"
            }
            
            await log_interaction_to_hotl(log_payload)
            
            self.query_one("#view-draft", Static).update(f"✅ Dispatched successfully:\n\n{log_payload['answer']}")
            self.query_one("#view-question", Static).update("Select a thread to view details.")
            self.query_one("#view-critique", Static).update("N/A")
            self.query_one("#history-box", Static).update("No active thread history loaded.")
            self.selected_thread_id = None
            
            await self.refresh_queue()
        except Exception as e:
            self.notify(f"Failed to resume graph thread: {str(e)}", severity="error")

if __name__ == "__main__":
    ReviewerDashboard().run()