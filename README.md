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
| `evals/` | evaluation harness: routing, full pipeline, retrieval and RAG generation (see [Evaluation](#evaluation)) |

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
GROQ_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...
GITHUB_TOKEN=...
```
`GROQ_API_KEY` selects Groq for the chat/tool-calling LLM (`ma_chat/config.py`, `LLM_MODEL`);
without it the agents fall back to an NVIDIA-hosted model. NVIDIA is always used for embeddings.
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

## Evaluation

This is a proof of concept running on free-tier APIs, so the evaluation sets are small and every figure below is
reported as `correct/total` with a 95% Wilson confidence interval. All questions and labels were written by the
author (questions paraphrased, labels checked against the stored text). Results are from 2026-09-19; per-case output
is saved under `evals/results/` (gitignored).

### Results

| Area | Metric | Result | What it measures |
|---|---|---|---|
| **Routing** (59 questions) | Overall accuracy | **56/59 = 94.9%** (CI 86-98%) | Did the orchestrator pick the right agent (MATH, RAG, GITHUB, TAVILY)? |
| | Easy questions | 40/40 = 100% | Unambiguous phrasing. Every model tried scored ~100% here, so it cannot tell routers apart. |
| | Hard questions, strict | 16/19 = 84.2% (CI 62-94%) | Keyword traps ("summarize the news", "calculate open issues"), typos, prompt injection, and follow-ups that depend on earlier turns. |
| | Hard questions, lenient | 18/19 = 94.7% | Same set, also accepting a reasonable alternative route on genuinely ambiguous questions. |
| | Follow-up rewriter | 4/4 | Did the orchestrator rewrite "How many stars does it have?" into a standalone question that names the repo? |
| **Retrieval** (24 questions, 214 chunks, 4 documents) | hit@5 | **23/24 = 95.8%** (CI 80-99%) | Is the chunk containing the answer in the top 5 results? |
| | hit@3 / hit@1 | 20/24 = 83.3% / 13/24 = 54.2% | Same, for stricter cut-offs. |
| | MRR | 0.696 | Average of 1/rank of the correct chunk. |
| | Correct document ranked first | 24/24 | Cross-document confusion: none. Every miss was a neighbouring chunk in the right document. |
| **RAG generation** (24 answerable + 8 unanswerable) | Answer correctness | **24/24 = 100%** (CI 86-100%) | Does the answer match a reference answer (LLM judge)? |
| | Abstention | **8/8 = 100%** (CI 68-100%) | On questions the documents cannot answer, does the agent decline instead of inventing an answer? |
| | Faithfulness | 18/24 = 75% (CI 55-88%) | Is every claim supported by the retrieved text? A strict judge; roughly two-thirds of the 6 flags were genuine embellishment, the rest judge strictness. |
| **MATH pipeline** (12 questions) | First-pass correct | **12/12** | Answer checked by code, not by a model. |
| | Effect of the `sp.sp.solve` fix | 4/6 to 6/6 first-pass; human escalations 3/6 to 0/6 | Same 6 questions before and after fixing a bug in `math_server.py` that the harness exposed. |
| | Cost and latency | ~950 tokens/query (2 generator calls + 1 Gemini call); p50 5.6 s, p95 6.6 s | Per-query LLM cost and end-to-end time. |

### Setup behind each result

| Result | Generator | Judge / evaluator | Embedding |
|---|---|---|---|
| Routing, MATH pipeline | `openai/gpt-oss-120b` (Groq) | in-loop evaluator `gemini-3.1-flash-lite` (routing never calls it) | - |
| Retrieval | - | - | `nvidia/nemotron-3-embed-1b` (2048-d, pgvector cosine) |
| RAG generation | `openai/gpt-oss-20b` (Groq) | `qwen/qwen3.8-27b` (a different model family) | `nvidia/nemotron-3-embed-1b` |

The RAG generation run used the smaller 20B model because the 120B model's free-tier daily token cap was used up by
earlier runs.

### What was not measured
- **Evaluator and retry effect.** The Gemini evaluator rejected the failing answers it saw, but retries fixed none of
  them and the sample is far too small to quote an improvement or an escalation rate.
- **GITHUB and TAVILY end to end.** Only routing is covered. The pipeline runs were blocked by an invalid GitHub token
  and API rate limits.
- **SUMMARY_AGENT** is deliberately excluded from all figures.
- **Tables in documents.** Not tested, and DOCX tables are not ingested at all (`load_docx` reads paragraphs only).

### Limitations
- Small samples (59 routing, 24 retrieval, 32 RAG generation questions) give wide intervals; treat the numbers as
  indicative, not as benchmarks.
- The MATH figures are a best case: they are the easiest category to grade, and the bug fix was made after seeing those
  failures.
- LLM judges are noisy. The faithfulness figure is a lower bound, and "cited only the correct document" is not
  reported because the agent cites all five retrieved chunks, so it mostly restates retrieval quality.
- Known ingestion weakness: `pypdf` drops the spaces in some PDFs (the HR handbook), which degrades its embeddings.

### Reproducing
Run from the repository root with the project's virtual environment activated.

```
python -m evals.run_eval --mode route                    # routing accuracy (no database needed)
python -m evals.run_eval --mode full --category MATH     # full graph: pipeline outcomes, cost, latency
python -m evals.run_eval --mode retrieval                # hit@k / MRR over evals/retrieval_cases.jsonl
python -m evals.rag_eval                                 # faithfulness, correctness, abstention
python -m evals.aggregate FILE.json ... --exclude SUMMARY_AGENT --regrade   # merge runs into one report
```

Case files are generated by `evals/make_cases.py` and `evals/make_retrieval_cases.py`; see `evals/README.md` for the
field reference. Each run needs the API keys above; the free tiers cap daily requests, so use `--per-category` or
`--limit` for cheap smoke tests.
