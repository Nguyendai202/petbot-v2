from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from openai import AsyncOpenAI

from vetbot import VetRetriever, stream_answer

# ── Init ──────────────────────────────────────────────────────────────────────

retriever = VetRetriever()
client = AsyncOpenAI()

# ── Persistence + Auth ────────────────────────────────────────────────────────

cl_data._data_layer = SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///data/chat_history.db")


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username == os.getenv("APP_USER", "admin") and password == os.getenv("APP_PASSWORD", "admin"):
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

    response_msg = cl.Message(content="")
    await response_msg.send()

    full_response = ""
    try:
        async for token in stream_answer(retriever, history, client):
            full_response += token
            await response_msg.stream_token(token)
    except Exception as exc:
        full_response = f"Đã có lỗi xảy ra: {exc}"

    response_msg.content = full_response
    await response_msg.update()

    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
