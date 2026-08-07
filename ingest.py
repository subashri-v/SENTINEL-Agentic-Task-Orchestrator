import argparse
import asyncio
import selectors

from ma_chat import config, database


async def main(filepaths):
    await database.init_db()
    try:
        for filepath in filepaths:
            await database.ingest_document(filepath)
    finally:
        if config.conn:
            await config.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and embed .pdf/.docx files into the RAG store.")
    parser.add_argument("files", nargs="+", help="Path(s) to .pdf or .docx files to ingest")
    args = parser.parse_args()

    asyncio.run(
        main(args.files),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
    )
