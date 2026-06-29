from dotenv import load_dotenv
load_dotenv()

import logging
import os
from typing import Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from openai import AsyncOpenAI

from vetbot import VetRetriever, stream_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Init ──────────────────────────────────────────────────────────────────────

retriever = VetRetriever()
client = AsyncOpenAI()

# ── Persistence + Auth ────────────────────────────────────────────────────────

_db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chat_history.db")

if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif _db_url.startswith("postgresql://") and "+asyncpg" not in _db_url:
    _db_url = _db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

cl_data._data_layer = SQLAlchemyDataLayer(conninfo=_db_url)


_APP_USER = os.environ["APP_USER"]
_APP_PASSWORD = os.environ["APP_PASSWORD"]


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username == _APP_USER and password == _APP_PASSWORD:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Xin chào! Tôi là trợ lý thú y. Hãy hỏi tôi về bệnh chó mèo.").send()


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    history = []
    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            history.append({"role": "user", "content": step.get("output", "")})
        elif step.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": step.get("output", "")})
    cl.user_session.set("history", history)


# ── Message handler ───────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    history: list = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    response_msg = cl.Message(content="*thinking...*")
    await response_msg.send()

    full_response = ""
    first_token = True

    try:
        async for item in stream_answer(retriever, history, client):

            # ── Thinking indicator — update the message text ─────────────
            if isinstance(item, dict) and item.get("type") == "thinking":
                response_msg.content = f"*{item['content']}*"
                await response_msg.update()
                continue

            # ── First token → clear thinking, start streaming ────────────
            if first_token:
                response_msg.content = ""
                first_token = False

            full_response += item
            await response_msg.stream_token(item)

    except Exception as exc:
        full_response = f"Đã có lỗi xảy ra: {exc}"

    response_msg.content = full_response
    await response_msg.update()

    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
