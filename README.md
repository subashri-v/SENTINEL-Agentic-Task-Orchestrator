# Agentic System Orchestrator

An async multi-agent chat system built on LangGraph. A single orchestrator node
routes each question to a specialist agent, a Gemini-based evaluator judges the
answer, and low-quality answers either retry or escalate to a human reviewer
(HITL).

## Architecture

```
User question
      |
      v
 orchestrator  --(rewrites pronoun-heavy queries into standalone questions,
      |           picks a route)
      |
      +--> MATH           (SymPy calculations via an MCP subprocess)
      +--> RAG             (pgvector similarity search over ingested documents)
      +--> TAVILY          (live web search)
      +--> GITHUB          (repo / issues lookup via PyGithub)
      +--> SUMMARY_AGENT   (summarizes the current chat thread)
      |
      v
  evaluator (Gemini judge)
      |
      +--> ACCEPT   -> done
      +--> retry route (up to 2 retries, with critique fed back in)
      +--> INTERVENT -> human_intervention (pauses the graph for a human answer)
```

Conversation/checkpoint state is persisted per `thread_id` in `checkpoints.db`
(SQLite, via LangGraph's `AsyncSqliteSaver`), which is what lets a thread pause
on human intervention and resume later. Document chunks/embeddings and an
audit log of every routed Q&A live in PostgreSQL.

## Project layout

| Path | Purpose |
|---|---|
| `ma_chat/config.py` | env/API key loading, model clients, DB config |
| `ma_chat/state.py` | `AgentState`, evaluator output schema |
| `ma_chat/database.py` | Postgres init, document loading/chunking, embeddings, retrieval, audit logging |
| `ma_chat/nodes.py` | the agent nodes (math, RAG, tavily, github, summary, orchestrator, evaluator, HITL) |
| `ma_chat/graph.py` | LangGraph wiring (`workflow`), `get_app`/`build_app` |
| `ma_chat/chat.py` | the terminal chat loop and app entrypoint |
| `async_ma_chat.py` | thin compatibility shim re-exporting the above + `if __name__ == "__main__"` runner |
| `math_server.py` | MCP server exposing a SymPy calculation tool to the MATH agent |
| `ingest.py` | CLI to chunk + embed a `.pdf`/`.docx` file into the RAG store |
| `async_cli.py` | Textual dashboard for end users to submit queries and watch threads |
| `async_rev.py` | Textual dashboard for reviewers to resolve HITL-escalated threads |
| `test.py` / `test_env.py` | one-off connectivity checks (NVIDIA API, Postgres logging) |

## Setup

### 1. Install Python dependencies
```
pip install -r requirements.txt
```

### 2. PostgreSQL + pgvector
You need a running PostgreSQL server with the `pgvector` extension available
(`CREATE EXTENSION vector;` must succeed). On Windows this extension has no
official prebuilt binary — build it from source against your PostgreSQL
install using the **x64 Native Tools Command Prompt for VS**:
```
set "PGROOT=C:\Program Files\PostgreSQL\<version>"
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```
Connection settings (host/port/dbname/user/password) are defined in
`ma_chat/config.py` (`DB_CONFIG`) — edit them there if your setup differs.
The app creates its own tables (`document_chunks`, `hotl_audit_logs`) on
startup, so no manual schema setup is needed once the extension is installed.

### 3. Environment variables
Create a `.env` file in the project root with:
```
NVIDIA_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...
GITHUB_TOKEN=...
```
`TAVILY_API_KEY` and `GITHUB_TOKEN` are optional — those agents report a
"not configured" message instead of failing if the key is missing.

## Running it

**Ingest documents into the RAG store:**
```
python ingest.py path/to/file.pdf path/to/other.docx
```

**Chat from the terminal:**
```
python async_ma_chat.py
```

**End-user query dashboard (Textual UI):**
```
python async_cli.py
```

**Reviewer dashboard for HITL escalations:**
```
python async_rev.py
```

You don't need to run `math_server.py` yourself — it's launched automatically
as an MCP subprocess whenever the MATH agent runs.
