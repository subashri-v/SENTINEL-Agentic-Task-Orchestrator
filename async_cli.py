import os
import sqlite3
import uuid
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, ListView, ListItem, Label, Input, Button, Static
from textual.reactive import reactive

# Import your State blueprint and compiled LangGraph pipeline instance
from async_ma_chat import AgentState, log_interaction_to_hotl
from langchain_core.messages import HumanMessage

from async_ma_chat import build_app

app = None

async def on_mount(self):
    global app
    app = await build_app()
    await self.update_dashboards()

class QueryClientDashboard(App):
    active_threads = reactive([])
    historical_tickets = reactive([])
    current_chat_thread_id = None
    
    CSS = """
    Screen {
        background: #111e25;
    }
    .panel-title {
        background: #1c2d37;
        color: #00d7ff;
        text-align: center;
        text-style: bold;
        padding: 1;
        margin-bottom: 1;
    }
    #input-pane {
        width: 45%;
        border-right: solid #1c2d37;
        padding: 1;
    }
    #history-pane {
        width: 55%;
        padding: 1;
    }
    .section-label {
        color: #ffaa00;
        text-style: bold;
        margin-top: 1;
    }
    .box-display {
        background: #17242c;
        border: solid #2c3e4b;
        min-height: 5;
        padding: 1;
        margin-bottom: 1;
    }
    #user-query-input {
        border: solid #00d7ff;
        margin-bottom: 1;
    }
    #submit-query-btn {
        background: #006688;
        color: white;
        margin-bottom: 2;
    }
    """
    
    pending_tickets = reactive([])
    historical_tickets = reactive([])

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            # Pane 1: User Query Desks & Active Intercept Monitors
            with Vertical(id="input-pane"):
                yield Label("✍️ ENTER NEW QUERY", classes="panel-title")
                yield Input(placeholder="Type your question here...", id="user-query-input")
                yield Button("Submit to Agent Network", id="submit-query-btn", variant="primary")
                
                yield Label("⏳ PENDING ESCALATION / HUMAN REVIEW QUEUE", classes="section-label")
                yield ListView(id="pending-list")
                yield Button("🔄 Sync / Refresh Session Status", id="sync-btn")

            # Pane 2: Complete Consolidated History Panel
            with Vertical(id="history-pane"):
                yield Label("📜 CONSOLIDATED HISTORIC LOGS & REPLIES", classes="panel-title")
                yield ListView(id="history-list")
                yield Label("Selected Thread Conversation Viewer:", classes="section-label")
                yield Static("Highlight an entry above to expand conversation logs.", id="chat-viewer-box", classes="box-display")

        yield Footer()

    async def on_mount(self) -> None:
        self.title = "LangGraph End-User Query Workspace"
        await self.update_dashboards()

async def update_dashboards(self) -> None:
        """Inspects checkpoints.db asynchronously to sort threads into running/review lanes vs closed lanes."""
        try:
            def _get_checkpoints():
                with sqlite3.connect("checkpoints.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
                    return [row[0] for row in cursor.fetchall()]

            all_threads = await asyncio.to_thread(_get_checkpoints)

            review_lane = []
            final_lane = []

            for tid in all_threads:
                config = {"configurable": {"thread_id": tid}}
                snapshot = await self.graph_app.get_state(config)
                
                if not snapshot.values.get("messages"):
                    continue

                if snapshot.next and "human_intervention" in snapshot.next:
                    review_lane.append(tid)
                else:
                    final_lane.append(tid)

            self.pending_tickets = review_lane
            self.historical_tickets = final_lane

            # Render Left Panel: Pending HITL Queue
            p_list = self.query_one("#pending-list", ListView)
            for item in p_list.query(ListItem):
                item.remove()
            for tid in self.pending_tickets:
                # FIX: Removed invalid 'await'
                p_list.append(ListItem(Label(f"⏳ Awaiting Approval: ...{tid[-12:]}"), id=f"pend-{tid}"))

            # Render Right Panel: Completed / Addressed Queue
            h_list = self.query_one("#history-list", ListView)
            for item in h_list.query(ListItem):
                item.remove()
            for tid in self.historical_tickets:
                # FIX: Removed invalid 'await'
                h_list.append(ListItem(Label(f"✅ Resolved / Clear: ...{tid[-12:]}"), id=f"hist-{tid}"))

        except Exception as e:
            self.notify(f"Database Refresh Interrupted: {str(e)}", severity="error")
import os
import sqlite3
import uuid
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, ListView, ListItem, Label, Input, Button, Static
from textual.reactive import reactive

# Import your State blueprint and compiled LangGraph pipeline instance
from async_ma_chat import AgentState, log_interaction_to_hotl
from langchain_core.messages import HumanMessage

from async_ma_chat import build_app, workflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class QueryClientDashboard(App):
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
    #input-pane {
        width: 50%;
        border-right: solid #333333;
        padding: 1;
    }
    #history-pane {
        width: 50%;
        padding: 1;
    }
    .section-label {
        color: #ffaa00;
        text-style: bold;
        margin-top: 1;
    }
    .box-display {
        background: #262626;
        border: solid #444444;
        min-height: 10;
        padding: 1;
        margin-top: 1;
    }
    """

    pending_tickets = reactive([])
    historical_tickets = reactive([])
    current_chat_thread_id = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            # Pane 1: User Query Desks & Active Intercept Monitors
            with Vertical(id="input-pane"):
                yield Label("✍️ ENTER NEW QUERY", classes="panel-title")
                yield Input(placeholder="Type your question here...", id="user-query-input")
                yield Button("Submit to Agent Network", id="submit-query-btn", variant="primary")
                
                yield Label("⏳ PENDING ESCALATION / HUMAN REVIEW QUEUE", classes="section-label")
                yield ListView(id="pending-list")
                yield Button("🔄 Sync / Refresh Session Status", id="sync-btn")

            # Pane 2: Complete Consolidated History Panel
            with Vertical(id="history-pane"):
                yield Label("📜 CONSOLIDATED HISTORIC LOGS & REPLIES", classes="panel-title")
                yield ListView(id="history-list")
                yield Label("Selected Thread Conversation Viewer:", classes="section-label")
                yield Static("Highlight an entry above to expand conversation logs.", id="chat-viewer-box", classes="box-display")

        yield Footer()

    async def on_unmount(self) -> None:
        """Cleanly tears down database connections on exit."""
        if hasattr(self, 'cp_context'):
            await self.cp_context.__aexit__(None, None, None)

    async def on_mount(self) -> None:
        self.title = "LangGraph End-User Query Workspace"
        
        # FIX: Dynamically wire up the checkpointer connection pool within the class lifecycle
        self.cp_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
        cp = await self.cp_context.__aenter__()
        
        # FIX: Save graph instance to self.graph_app to prevent Textual read-only app attribute clash
        self.graph_app = workflow.compile(checkpointer=cp)
        
        await self.update_dashboards()

    async def update_dashboards(self) -> None:
        """Inspects checkpoints.db asynchronously to sort threads into running/review lanes vs closed lanes."""
        try:
            def _get_checkpoints():
                with sqlite3.connect("checkpoints.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
                    return [row[0] for row in cursor.fetchall()]

            all_threads = await asyncio.to_thread(_get_checkpoints)

            review_lane = []
            final_lane = []

            for tid in all_threads:
                config = {"configurable": {"thread_id": tid}}
                
                # FIX: Changed get_state to async aget_state to prevent thread blocking error
                snapshot = await self.graph_app.aget_state(config)
                
                if not snapshot.values.get("messages"):
                    continue

                if snapshot.next and "human_intervention" in snapshot.next:
                    review_lane.append(tid)
                else:
                    final_lane.append(tid)

            self.pending_tickets = review_lane
            self.historical_tickets = final_lane

            # Render Left Panel: Pending HITL Queue
            p_list = self.query_one("#pending-list", ListView)
            for item in p_list.query(ListItem):
                item.remove()
            for tid in self.pending_tickets:
                p_list.append(ListItem(Label(f"⏳ Awaiting Approval: ...{tid[-12:]}"), id=f"pend-{tid}"))

            # Render Right Panel: Completed / Addressed Queue
            h_list = self.query_one("#history-list", ListView)
            for item in h_list.query(ListItem):
                item.remove()
            for tid in self.historical_tickets:
                h_list.append(ListItem(Label(f"✅ Resolved / Clear: ...{tid[-12:]}"), id=f"hist-{tid}"))

        except Exception as e:
            self.notify(f"Database Refresh Interrupted: {str(e)}", severity="error")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit-query-btn":
            await self.fire_user_query()
        elif event.button.id == "sync-btn":
            await self.update_dashboards()
            self.notify("Session Sync Completed.")

    async def fire_user_query(self) -> None:
        input_widget = self.query_one("#user-query-input", Input)
        query_text = input_widget.value.strip()

        if not query_text:
            self.notify("Cannot submit an empty query prompt.", severity="warning")
            return

        if not self.current_chat_thread_id:
            self.current_chat_thread_id = str(uuid.uuid4())
            
        config = {"configurable": {"thread_id": self.current_chat_thread_id}}
        
        payload = {
            "question": query_text,
            "messages": [HumanMessage(content=query_text)],
            "retry_count": 0
        }

        self.notify(f"Dispatched to Thread ...{self.current_chat_thread_id[-12:]}")
        input_widget.value = "" 

        final_state = None

        try:
            # FIX: Point stream events to self.graph_app
            async for event in self.graph_app.astream_events(payload, version="v2", config=config):
                # Capture FINAL GRAPH STATE ONLY
                if event["event"] == "on_chain_end" and "output" in event["data"]:
                    final_state = event["data"]["output"]

            # Fallback safety
            if final_state is None:
                snapshot = await self.graph_app.get_state(config)
                final_state = snapshot.values

            try:
                await log_interaction_to_hotl(final_state)
            except Exception as pg_log_err:
                # Keep client running even if the postgres sync blips
                self.notify(f"Audit log sync failed: {pg_log_err}", severity="warning")

            snapshot = await self.graph_app.get_state(config)

            if snapshot.next:
                self.notify("⚠️ Verification failed. Routed to Human Supervisor Desk.", severity="warning")
                self.current_chat_thread_id = None
            else:
                self.notify("Response finalized natively.", severity="success")

            # FIX: Force immediate state sync
            await self.update_dashboards()

            if self.current_chat_thread_id:
                await self.render_chat_log(self.current_chat_thread_id)

        except Exception as e:
            self.notify(f"Execution Fault: {e}", severity="error")

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item and event.item.id:
            clean_thread_id = event.item.id.replace("pend-", "").replace("hist-", "")
            self.current_chat_thread_id = clean_thread_id 
            await self.render_chat_log(clean_thread_id)

    async def render_chat_log(self, thread_id: str) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        
        # FIX: Changed get_state to async aget_state
        snapshot = await self.graph_app.aget_state(config)
        messages = snapshot.values.get("messages", [])

        conversation_markup = []
        for msg in messages:
            role = "👤 User" if msg.type == "human" else "🤖 Agent"
            conversation_markup.append(f"[b]{role}:[/b]\n{msg.content}\n")

        if snapshot.next and "human_intervention" in snapshot.next:
            conversation_markup.append("[yellow][i]⏳ Currently held under review by administrator desk...[/i][/yellow]")

        log_string = "\n".join(conversation_markup) if conversation_markup else "No session messages located."
        self.query_one("#chat-viewer-box", Static).update(log_string)

    async def start_new_topic(self) -> None:
        self.current_chat_thread_id = None
        self.query_one("#chat-viewer-box", Static).update("Started fresh session. Type your query below.")
        self.notify("Fresh conversation sequence initialized.")


if __name__ == "__main__":
    QueryClientDashboard().run()