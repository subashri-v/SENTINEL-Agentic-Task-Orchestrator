import os
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= NVIDIA_API_KEY
)

async def test():
    r = await client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role":"user","content":"Hello"}]
    )
    print(r.choices[0].message.content)

asyncio.run(test())