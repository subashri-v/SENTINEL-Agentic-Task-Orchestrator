import os
import asyncio

import psycopg
from pgvector.psycopg import register_vector_async  # Updated for psycopg v3

from pypdf import PdfReader
from docx import Document

from langchain_mcp_adapters.client import MultiServerMCPClient

from . import config


async def init_db():
    config.conn = await psycopg.AsyncConnection.connect(**config.DB_CONFIG)

    async with config.conn.cursor() as cur:
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await config.conn.commit()

    # IMPORTANT: register AFTER extension exists
    await register_vector_async(config.conn)

    async with config.conn.cursor() as cur:
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

        await config.conn.commit()


async def init_db_and_mcp():
    # 1. First run the original database creation logic
    await init_db()

    # 2. Wire up the MultiServerMCPClient mapping
    config.mcp_client = MultiServerMCPClient({
        "mathengine": {
            "command": "python",
            "args": [config.MATH_SERVER_PATH],
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

    async with config.conn.cursor() as cur:
        for idx, chunk in enumerate(chunks):
            embedding = await get_embedding(chunk, is_query=False)
            await cur.execute("""
                INSERT INTO document_chunks (source_file, chunk_number, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (os.path.basename(filepath), idx, chunk, embedding))
        await config.conn.commit()
    print(f"Stored {len(chunks)} chunks")

async def get_embedding(text, is_query=False):
    if not text or not text.strip():
        return [0.0] * 1024
    input_type = "query" if is_query else "passage"
    response = await config.nvidia_client.embeddings.create(
        input=[text],
        model=config.EMBEDDING_MODEL,
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return response.data[0].embedding

async def retrieve(question):
    q_embedding = await get_embedding(question, is_query=True)
    async with config.conn.cursor() as cur:
        await cur.execute("""
            SELECT source_file, chunk_number, content FROM document_chunks
            ORDER BY embedding <=> %s::vector LIMIT %s
        """, (q_embedding, config.TOP_K))
        return await cur.fetchall()

async def log_interaction_to_hotl(log_payload: dict):
    try:
        connection_str = (
            config.DB_CONFIG
            if isinstance(config.DB_CONFIG, str)
            else f"dbname={config.DB_CONFIG.get('dbname')} user={config.DB_CONFIG.get('user')} password={config.DB_CONFIG.get('password')} host={config.DB_CONFIG.get('host')} port={config.DB_CONFIG.get('port', 5432)}"
        )

        # FIX: Check if global async connection exists
        if config.conn is None:
            # Open a standalone temporary async connection just for this log entry
            async with await psycopg.AsyncConnection.connect(connection_str) as temp_conn:
                async with temp_conn.cursor() as cursor:
                    await _execute_insert_query(cursor, log_payload)
        else:
            # FIX: Use 'async with' context manager for the active async connection cursor
            async with config.conn.cursor() as cursor:
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
