import importlib.resources.abc
import importlib.abc
import sys

# Also inject it directly into the sys.modules instance if needed
sys.modules['importlib.abc'].Traversable = importlib.resources.abc.Traversable

# 2. NOW import the rest of your libraries
import os
import json
import asyncio
import sqlite3
import uuid
from dotenv import load_dotenv

# Async DB Driver & Helpers
import psycopg
from pgvector.psycopg import register_vector_async  # Updated for psycopg v3

# Document Parsers
from pypdf import PdfReader
from docx import Document

# Model APIs
from openai import AsyncOpenAI
import google.generativeai as genai

# LangGraph Engine
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # Changed to Async version
from langchain_mcp_adapters.client import MultiServerMCPClient

MATH_SERVER_PATH = os.path.abspath("math_server.py")
mcp_client = None

# SDK Clients
from tavily import TavilyClient
from github import Github

load_dotenv()
app=None

# API Keys Initialization
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

print("--- API Key Status ---")
print("NVIDIA_API_KEY:", "Found" if NVIDIA_API_KEY else "Not Found")
print("GEMINI_API_KEY:", "Found" if GEMINI_API_KEY else "Not Found")
print("TAVILY_API_KEY:", "Found" if TAVILY_API_KEY else "Not Found")
print("GITHUB_TOKEN:", "Found" if GITHUB_TOKEN else "Not Found")

# Gemini Configurations
GEMINI_MODEL = "gemini-2.5-flash"
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
gemini = genai.GenerativeModel(GEMINI_MODEL)

# Database Parameters
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ragdb",  # psycopg3 prefers dbname over database
    "user": "postgres",
    "password": "12345"
}

TOP_K = 5
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Keep standard clients for threading offload wrappers
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Global placeholder connection object
conn = None

async def init_db():
    global conn

    conn = await psycopg.AsyncConnection.connect(**DB_CONFIG)

    async with conn.cursor() as cur:
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.commit()

    # IMPORTANT: register AFTER extension exists
    from pgvector.psycopg import register_vector_async
    await register_vector_async(conn)

    async with conn.cursor() as cur:
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source_file TEXT,
                chunk_number INT,
                content TEXT,
                embedding vector(1024)
            );
        """)

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS hotl_audit_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                question TEXT,
                answer TEXT,
                final_route TEXT,
                retry_count INT,
                system_critique TEXT
            );
        """)

        await conn.commit()

async def init_db_and_mcp():
    global mcp_client
    # 1. First run your original database creation logic
    await init_db() 
    
    # 2. Wire up the MultiServerMCPClient mapping
    mcp_client = MultiServerMCPClient({
        "mathengine": {
            "command": "python",
            "args": [MATH_SERVER_PATH],
            "transport": "stdio"
        }
    })

# =====================================================
# DOCUMENT LOADERS & UTILITIES
# =====================================================
async def load_pdf(path):
    # Offloading synchronous file I/O tracking to threads
    def _read():
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt: pages.append(txt)
        return "\n".join(pages)
    return await asyncio.to_thread(_read)

async def load_docx(path):
    def _read():
        doc = Document(path)
        return "\n".join(para.text for para in doc.paragraphs)
    return await asyncio.to_thread(_read)

async def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return await load_pdf(path)
    if ext == ".docx": return await load_docx(path)
    raise Exception(f"Unsupported file type: {ext}")

async def chunk_text(text, chunk_size=700, overlap=150):
    if not text or not text.strip(): return []
    chunks = []
    start = 0
    text_len = len(text)
    step = chunk_size - overlap
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len: break
        start += step
    return chunks

async def ingest_document(filepath):
    print(f"\nIngesting: {filepath}")
    text = await load_document(filepath)
    chunks = await chunk_text(text)
    if not chunks: return
    
    async with conn.cursor() as cur:
        for idx, chunk in enumerate(chunks):
            embedding = await get_embedding(chunk, is_query=False)
            await cur.execute("""
                INSERT INTO document_chunks (source_file, chunk_number, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (os.path.basename(filepath), idx, chunk, embedding))
        await conn.commit()
    print(f"Stored {len(chunks)} chunks")

async def get_embedding(text, is_query=False):
    if not text or not text.strip(): 
        return [0.0] * 1024
    input_type = "query" if is_query else "passage"
    response = await nvidia_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return response.data[0].embedding

async def retrieve(question):
    q_embedding = await get_embedding(question, is_query=True) 
    async with conn.cursor() as cur:
        await cur.execute("""
            SELECT source_file, chunk_number, content FROM document_chunks
            ORDER BY embedding <=> %s::vector LIMIT %s
        """, (q_embedding, TOP_K))
        return await cur.fetchall()

# Inside async_ma_chat.py
async def log_interaction_to_hotl(log_payload: dict):
    global conn  # Refers to your active async postgres connection
    
    try:
        connection_str = (
            DB_CONFIG 
            if isinstance(DB_CONFIG, str) 
            else f"dbname={DB_CONFIG.get('dbname')} user={DB_CONFIG.get('user')} password={DB_CONFIG.get('password')} host={DB_CONFIG.get('host')} port={DB_CONFIG.get('port', 5432)}"
        )
        
        # FIX: Check if global async connection exists
        if conn is None:
            # Open a standalone temporary async connection just for this log entry
            async with await psycopg.AsyncConnection.connect(connection_str) as temp_conn:
                async with temp_conn.cursor() as cursor:
                    await _execute_insert_query(cursor, log_payload)
        else:
            # FIX: Use 'async with' context manager for the active async connection cursor
            async with conn.cursor() as cursor:
                await _execute_insert_query(cursor, log_payload)
                
    except Exception as e:
        print(f"[HOTL Error] Failed to write log: {e}")
        raise e 

async def _execute_insert_query(cursor, payload):
    """Helper that explicitly matches your hotl_audit_logs PostgreSQL schema columns and commits."""
    query = """
        INSERT INTO hotl_audit_logs (question, answer, final_route, retry_count, system_critique)
        VALUES (%s, %s, %s, %s, %s);
    """
    
    # Extract the raw text out of the list-of-dicts response formatting if needed
    raw_answer = payload.get("answer")
    if isinstance(raw_answer, list) and len(raw_answer) > 0:
        if isinstance(raw_answer[0], dict) and "text" in raw_answer[0]:
            raw_answer = raw_answer[0]["text"]
            
    # 1. Execute the query
    await cursor.execute(query, (
        payload.get("question"),
        str(raw_answer),
        payload.get("next_agent") or payload.get("final_route"),
        payload.get("retry_count", 0),
        payload.get("system_critique")
    ))
    
    # 2. FIX: Explicitly commit the transaction block immediately 
    # to push the data out of the memory buffer and into the physical table.
    await cursor.connection.commit()

# =====================================================
# LANGGRAPH STATE DEFINITION
# =====================================================
from typing import List, Annotated
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    next_agent: str
    answer: str
    citations: List[str]
    retry_count: int
    system_critique: str

# =====================================================
# THE NODES
# =====================================================

async def node_math_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running MCP Math Engine Agent (NVIDIA)...")
    question = state["question"]
    
    # 1. FIX: Await the tools directly from the multi-server adapter
    lc_tools = await mcp_client.get_tools() 
    
    # 2. Convert LangChain tool wrappers into OpenAI tools format for the NVIDIA client
    from langchain_core.utils.function_calling import convert_to_openai_tool
    openai_tools = [convert_to_openai_tool(t) for t in lc_tools]
    
    prompt = (
        "You are a calculation router bridging to a SymPy mathematical engine.\n"
        "You must pass complex calculations or algebraic strings to your calculation engine tool.\n"
        "CRITICAL SYNTAX RULES:\n"
        "- Use standard Python operators (e.g., use x**2 for squaring, not x^2).\n"
        "- For derivatives, use 'sp.diff(expression, x)'.\n"
        "- For equations or finding roots, use 'sp.solve(expression, x)'.\n\n"
        f"User Query: {question}"
    )
    
    # 3. Request inference with converted tools structure bound
    response = await nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=openai_tools,
        timeout=60
    )
    
    message = response.choices[0].message
    
    # 4. Execute tool call if requested by Llama 3.1
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print(f"[Math Agent] Model invoking MCP tool: {tool_name} with args: {tool_args}")
        
        # Cross-reference the selected function name against your LangChain tools pool
        target_tool = next(t for t in lc_tools if t.name == tool_name)
        execution_output = await target_tool.ainvoke(tool_args)
        
        final_answer = str(execution_output)
        return {
            "answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
            "citations": ["Internal MCP MathEngine Subprocess"],
            "next_agent": "MATH"
        }
        
    ai_answer = message.content
    return {
        "answer": ai_answer,
        "messages": [AIMessage(content=ai_answer)],
        "citations": [],
        "next_agent": "MATH"
    }
        

async def node_rag_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running RAG Agent (NVIDIA)...")
    question = state["question"]
    rows = await retrieve(question)
    
    if not rows:
        err_msg = "I couldn't find matching info in your documents."
        return {
            "answer": err_msg, 
            "messages": [AIMessage(content=err_msg)], 
            "citations": [], 
            "next_agent": "RAG"
        }

    context_parts = []
    citations = []
    for row in rows:
        citations.append(f"{row[0]} (Chunk {row[1]})")
        context_parts.append(f"SOURCE: {row[0]}\n\n{row[2]}")

    context_text = "\n\n".join(context_parts)
    prompt = f"Answer from context ONLY.\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"
    
    response = await nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_answer = response.choices[0].message.content
    return {
        "answer": ai_answer, 
        "messages": [AIMessage(content=ai_answer)], 
        "citations": list(set(citations)), 
        "next_agent": "RAG"
    }

async def node_tavily_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running Tavily Web Agent (NVIDIA)...")
    question = state["question"]
    
    if not tavily_client:
        err_msg = "Tavily Web Agent could not run because TAVILY_API_KEY is unconfigured."
        return {
            "answer": err_msg, 
            "messages": [AIMessage(content=err_msg)], 
            "citations": [],
            "next_agent": "TAVILY"
        }
        
    try:
        # Offload sync client engine call to threadpool loop
        search_result = await asyncio.to_thread(tavily_client.search, query=question, max_results=3)
        context_parts = [f"URL: {r['url']}\nContent: {r['content']}" for r in search_result.get("results", [])]
        citations = [r['url'] for r in search_result.get("results", [])]
        
        context_text = "\n\n".join(context_parts)
        prompt = f"Answer using the web data below concisely.\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"
        
        response = await nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        ai_answer = response.choices[0].message.content
        return {
            "answer": ai_answer, 
            "messages": [AIMessage(content=ai_answer)], 
            "citations": citations, 
            "next_agent": "TAVILY"
        }
    except Exception as e:
        err_ex = f"Tavily API Runtime Exception: {str(e)}"
        return {
            "answer": err_ex, 
            "messages": [AIMessage(content=err_ex)], 
            "citations": [], 
            "next_agent": "TAVILY"
        }

async def node_github_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running GitHub Agent (NVIDIA)...")
    question = state["question"]
    critique = state.get("system_critique", "")
    
    if not github_client:
        err_msg = "Missing GITHUB_TOKEN environment variable."
        return {
            "answer": err_msg, 
            "messages": [AIMessage(content=err_msg)], 
            "citations": [], 
            "next_agent": "GITHUB"
        }
        
    try:
        extractor = (
            "Extract 'owner/repo' format from query. Return ONLY that string. "
            "If the query is general or doesn't mention a specific repo, reply with 'NONE'.\n"
            f"Query: {question}"
        )
        
        repo_res = await nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": extractor}]
        )
        repo_name = repo_res.choices[0].message.content.strip() if repo_res.choices[0].message.content else "NONE"
        
        if "NONE" in repo_name or "/" not in repo_name:
            # Wrap PyGithub blocking API steps inside a helper method for thread scheduling
            def _get_user_repos():
                user = github_client.get_user()
                repos = user.get_repos()
                sample = [r.name for idx, r in enumerate(repos) if idx < 5]
                return user.login, repos.totalCount, sample, user.html_url
                
            login, total_repo_count, sample_repos, html_url = await asyncio.to_thread(_get_user_repos)
            
            prompt = f"""
            You are interacting with a user's authenticated GitHub account data.
            Account Owner Login Name: {login}
            Total Repositories on Account: {total_repo_count}
            Sample accessible repo names: {', '.join(sample_repos)}

            USER QUESTION: {question}
            PREVIOUS EVALUATOR CRITIQUE (IF ANY): {critique}

            Answer the user's question accurately. Make sure to adhere to any formatting instructions given in the question!
            """
            
            response = await nvidia_client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_answer = response.choices[0].message.content
            return {
                "answer": ai_answer, 
                "messages": [AIMessage(content=ai_answer)], 
                "citations": [html_url],
                "next_agent": "GITHUB"
            }
        
        # Offload single repository extraction logic
        def _get_repo_details(name):
            r = github_client.get_repo(name)
            issues = [f"#{i.number}: {i.title}" for idx, i in enumerate(r.get_issues(state="open")) if idx < 3]
            return r.full_name, r.description, issues, r.html_url
            
        full_name, description, issues, repo_url = await asyncio.to_thread(_get_repo_details, repo_name)
        issue_text = "\n".join(issues) if issues else "No open issues found."
        
        ctx = f"Repo: {full_name}\nDesc: {description}\nOpen Issues:\n{issue_text}"
        prompt = f"Answer using this repository context.\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{question}\nCRITIQUE: {critique}"
        
        response = await nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_answer = response.choices[0].message.content
        return {
            "answer": ai_answer, 
            "messages": [AIMessage(content=ai_answer)], 
            "citations": [repo_url],
            "next_agent": "GITHUB"
        }
        
    except Exception as e:
        err_ex = f"GitHub Agent caught an execution error: {str(e)}"
        return {
            "answer": err_ex, 
            "messages": [AIMessage(content=err_ex)], 
            "citations": [], 
            "next_agent": "GITHUB"
        }

async def node_summary_agent(state: AgentState) -> dict:
    print("\n[Node] -> Running Summary & Chat History Agent...")
    question = state["question"]
    
    history_lines = []
    for msg in state.get("messages", []):
        sender = "User" if msg.type == "human" else "Assistant"
        history_lines.append(f"{sender}: {msg.content}")
    history_str = "\n".join(history_lines)
    
    prompt = f"""You are an internal operations assistant with clear visibility into the ongoing conversation thread history.
    The user is asking you for a summary or has a question regarding the past sequence of interactions in this specific session.
    
    CURRENT CONVERSATION HISTORY LOG:
    {history_str}
    
    USER TASK:
    {question}
    
    Fulfill their request explicitly based on the logs above.
    """
    
    response = await nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_summary = response.choices[0].message.content
    return {
        "answer": ai_summary, 
        "messages": [AIMessage(content=ai_summary)], 
        "citations": [], 
        "next_agent": "SUMMARY_AGENT"
    }

async def node_orchestrator(state: AgentState) -> dict:
    print(f"\n[Orchestrator] Reviewing query: '{state['question']}'")
    history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-5:]])
    
    prompt = f"""You are a master routing supervisor and query rewriter.
    
    RECENT CONVERSATION HISTORY FOR CONTEXT:
    {history}

    ROUTING RULES:
    - Choose 'MATH' for computational problems, algebra, symbols, or equations.
    - Choose 'RAG' for internal files, uploaded documents, universities.
    - Choose 'GITHUB' for repository details, git issues, open PRs.
    - Choose 'TAVILY' for recent events, search engine lookups, live internet news.
    - Choose 'SUMMARY_AGENT' for summaries of this chat session.
    
    TASK:
    1. Select the correct agent.
    2. Analyze the Latest User Query. If it uses pronouns ("he", "it", "that position") or relies on past context, rewrite it into a completely standalone, explicit question. If it's already standalone, leave it as is.

    Provide your output strictly in valid JSON matching this schema:
    {{
        "chosen_agent": "MATH" |"RAG" | "GITHUB" | "TAVILY" | "SUMMARY_AGENT",
        "standalone_question": "The rewritten explicit question string",
        "reasoning": "string"
    }}

    Latest User Query: "{state['question']}"
    """
    
    response = await nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_content = response.choices[0].message.content

    if raw_content.startswith("```"):
        raw_content = raw_content.strip("```json").strip("```").strip()
    
    try:
        decision = json.loads(raw_content)
        agent = decision.get("chosen_agent", "TAVILY").upper()
        optimized_question = decision.get("standalone_question", state["question"])
    except Exception as parse_err:
        print(f"[Orchestrator Fallback] Parsing failed: {parse_err}. Raw content: {raw_content}")
        # Intelligent structural fallback
        if "MATH" in raw_content or "x**" in state["question"]:
            agent = "MATH"
        else:
            agent = "TAVILY"
        optimized_question = state["question"]
        
    print(f"[Orchestrator Decision] Route to: {agent}")
    print(f"[Orchestrator Rewriter] Optimized Query: '{optimized_question}'")
    return {"next_agent": agent, "question": optimized_question}

async def route_decision(state: AgentState) -> str:
    return state["next_agent"]

# =====================================================
# EVALUATION AGENT
# =====================================================
class EvalSchema(TypedDict):
    passed: bool
    reasoning: str

async def node_evaluator(state: AgentState) -> dict:
    print("\n[Evaluator] -> Judging with Gemini...")
    question = state["question"]
    answer = state["answer"]
    retry_count = state.get("retry_count", 0)
    current_route = state.get("next_agent", "TAVILY")

    if retry_count >= 2:
        print("[Evaluator Decision] Max retries reached. Handing over to HUMAN INTERVENTION.")
        return {"next_agent": "INTERVENT"}

    eval_prompt = f"""You are an objective AI quality judge...
    QUESTION: "{question}"
    ANSWER: "{answer}"
    """

    try:
        # Wrap the generative sync implementation safely into an external execution block
        def _call_gemini():
            return gemini.generate_content(
                eval_prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": EvalSchema
                }
            )
        response = await asyncio.to_thread(_call_gemini)
        result = json.loads(response.text)
        passed = result.get("passed", True)
        print(f"[Gemini Evaluator] Passed: {passed} | Reason: {result.get('reasoning')}")
        
        if passed:
            return {"next_agent": "ACCEPT"}
        else:
            return {
                "next_agent": current_route, 
                "retry_count": retry_count + 1,
                "system_critique": result.get("reasoning")
            }
            
    except Exception as e:
        print(f"[Evaluator Error] Gemini failed: {e}. Handing over to HUMAN INTERVENTION.")
        return {"next_agent": "INTERVENT"}

# =====================================================
# HITL (Human-In-The-Loop)
# =====================================================
async def node_human_intervention(state: AgentState):
    print(f"\n[HITL State] Yielding control to state store for Thread: {state.get('question')}")
    
    human_response = interrupt({
        "question": state["question"],
        "generated_answer": state["answer"],
        "critique": state.get("system_critique", "")
    })

    return {
        "answer": human_response,
        "messages": [AIMessage(content=human_response)],
        "next_agent": "ACCEPT"
    }
    
# =====================================================
# BUILDING THE LANGGRAPH PIPELINE
# =====================================================
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", node_orchestrator)
workflow.add_node("RAG", node_rag_agent)
workflow.add_node("TAVILY", node_tavily_agent)
workflow.add_node("GITHUB", node_github_agent)
workflow.add_node("SUMMARY_AGENT", node_summary_agent)
workflow.add_node("evaluator", node_evaluator)
workflow.add_node("human_intervention", node_human_intervention)
workflow.add_node("MATH", node_math_agent)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
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


def evaluate_decision(state: AgentState) -> str:
    return state["next_agent"]

workflow.add_conditional_edges(
    "evaluator",
    evaluate_decision,
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
        config = {"configurable": {"thread_id": thread_id}}
        
        payload = {
            "question": user_input, 
            "messages": [HumanMessage(content=user_input)],
            "retry_count": 0
        }
        
        print(f"Processing Request under Thread ID: {thread_id} ...")
        # Fixed: Changed from .invoke() to await .ainvoke()
        result = await app.ainvoke(payload, config=config)
        
        snapshot = await app.aget_state(config)
        if snapshot.next:
            print(f"⚠️ [Status] Thread {thread_id} requires supervisor approval.")
            print("Your query has been queued for evaluation. You may continue typing new questions.")
            continue
            
        print(f"\n[Answer]: {result.get('answer')}")
        await log_interaction_to_hotl(result)

        print("\n" + "="*20 + " CITATIONS " + "="*20)
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                print("-", c)
        else:
            print("No background documents or endpoints referenced.")

def get_app():
    """
    Synchronous fallback or retrieval. Note: Because AsyncSqliteSaver 
    requires an active async connection pool, compiling a fresh instance 
    with a long-lived connection is much safer for secondary tools like the UI.
    """
    global app
    if app is None:
        # Fallback setup: create a standard connection pool for the checkpointer
        # so it stays alive across multiple independent UI calls.
        from langgraph.checkpoint.postgres import AsyncPostgresSaver # if postgres
        # For AsyncSqliteSaver, we can create a persistent connection string instance
        cp = AsyncSqliteSaver.from_conn_string("checkpoints.db")
        app = workflow.compile(checkpointer=cp)
    return app

async def build_app():
    global app
    await init_db()
    # Create the checkpointer without the 'async with' auto-close block 
    # if you intend to reuse 'app' outside of this function scope.
    cp = AsyncSqliteSaver.from_conn_string("checkpoints.db")
    app = workflow.compile(checkpointer=cp)
    return app

async def main():
    # 1. Initialize PostgreSQL schemas & background MCP configurations
    await init_db_and_mcp()
    global app
    
    # 2. Extract and keep the checkpointer context persistent
    cp_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
    checkpointer = await cp_context.__aenter__()
    
    # 3. Compile the graph onto the global reference layer safely
    app = workflow.compile(checkpointer=checkpointer)
    
    print("\n[System] LangGraph Agent Pipeline Compiled Successfully.")
    
    try:
        # FIX: Call chat directly without the invalid 'async with mcp_client'
        await chat(app)
    finally:
        # 4. Clean up all persistent pools gracefully on hard application exits
        await cp_context.__aexit__(None, None, None)
        
        if conn:
            print("[System] Closing PostgreSQL primary connections...")
            await conn.close()


if __name__ == "__main__":
    import selectors
    
    # Configure asyncio to use SelectorEventLoop to satisfy psycopg async connections
    asyncio.run(
        main(), 
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
    )