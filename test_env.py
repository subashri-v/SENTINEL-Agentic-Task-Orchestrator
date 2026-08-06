import asyncio
from async_ma_chat import DB_CONFIG, log_interaction_to_hotl

async def test_log():
    print(f"Checking DB_CONFIG: {DB_CONFIG}")
    sample_payload = {
        "question": "Diagnostic Test Question",
        "answer": "Diagnostic Test Answer",
        "final_route": "ACCEPT",
        "retry_count": 0,
        "system_critique": "Testing connection from UI environment"
    }
    
    try:
        print("Attempting to write log to PostgreSQL...")
        await log_interaction_to_hotl(sample_payload)
        print("✅ SUCCESS: Row inserted successfully into your PostgreSQL audit logs table!")
    except Exception as e:
        print(f"❌ FAILED: PostgreSQL logger failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_log())