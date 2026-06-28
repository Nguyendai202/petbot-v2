import json
import logging
from typing import AsyncGenerator

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from .prompts import (
    SYSTEM as _SYSTEM,
    CLASSIFY_SYSTEM as _CLASSIFY_SYSTEM,
    QUERY_UNDERSTANDING_SYSTEM as _QUERY_UNDERSTANDING_SYSTEM,
    DISAMBIGUATE_SYSTEM as _DISAMBIGUATE_SYSTEM,
)
from .retriever import VetRetriever

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 10
_MAX_FOLLOWUP = 4
_MAX_DISAMBIGUATE = 2


def _trim(messages: list[dict]) -> list[dict]:
    return messages[-(_MAX_HISTORY_TURNS * 2):]


def _count_followup_turns(messages: list[dict]) -> int:
    count = 0
    for msg in messages:
        if msg["role"] == "assistant" and "?" in msg["content"]:
            count += 1
    return count


def _is_emergency(text: str) -> bool:
    keywords = [
        "co giật", "giật toàn thân", "liệt",
        "không thở", "tím lưỡi", "bất tỉnh", "hôn mê",
        "không tiểu được", "nôn ra máu",
        "tai nạn", "chấn thương nghiêm trọng"
    ]
    return any(kw in text.lower() for kw in keywords)


def _format_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages[-8:]:
        role = "Người dùng" if msg["role"] == "user" else "Chatbot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


async def _classify(messages, client, model) -> dict:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": _format_history(messages)},
            ],
            stream=False,
            response_format={"type": "json_object"},
            langsmith_extra={"name": "vet-classify"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"state": "ready", "followup_question": ""}


async def _query_understanding(messages, client, model) -> dict:
    """
    Only decides: chitchat or needs retrieval?
    If retrieval is needed, use conversation content directly as the query —
    don't rewrite it, to avoid introducing assumptions.
    """
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _QUERY_UNDERSTANDING_SYSTEM},
                {"role": "user", "content": _format_history(messages)},
            ],
            stream=False,
            response_format={"type": "json_object"},
            langsmith_extra={"name": "vet-query-understanding"},
        )
        result = json.loads(resp.choices[0].message.content)
        logger.info("query_understanding needs_retrieval=%s", result.get("needs_retrieval"))
        return result
    except Exception:
        return {"needs_retrieval": True, "chitchat_response": ""}


def _count_disambiguate_turns(messages: list[dict]) -> int:
    """Count how many disambiguation questions have been asked (based on content markers)."""
    return sum(
        1 for msg in messages
        if msg["role"] == "assistant" and "?" in msg["content"]
        and any(kw in msg["content"].lower() for kw in ["mắt", "phân", "nôn", "màu", "mùi", "lần"])
    )


async def _disambiguate(
    messages: list[dict],
    coarse_context: str,
    client,
    model: str,
) -> dict:
    """After coarse retrieve: decide whether there's enough info to commit to a disease.
    - confident=True  → {"confident": True, "disease": "Disease X", "question": ""}
    - confident=False → {"confident": False, "disease": "", "question": "a question"}
    """
    symptoms = _format_history(messages)
    prompt = f"TRIỆU CHỨNG:\n{symptoms}\n\nTÀI LIỆU THAM KHẢO:\n{coarse_context}"
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _DISAMBIGUATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            response_format={"type": "json_object"},
            langsmith_extra={"name": "vet-disambiguate"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"confident": True, "disease": "", "question": ""}


def _thinking(content: str) -> dict:
    """Build a thinking signal for app.py to show the user what's happening."""
    return {"type": "thinking", "content": content}


async def stream_answer(
    retriever: VetRetriever,
    messages: list[dict],
    client: AsyncOpenAI,
    routing_model: str = "gpt-4o-mini",
    diagnosis_model: str = "gpt-4o",
) -> AsyncGenerator[str, None]:
    """
    Flow:
    1. Emergency check → immediate first aid
    2. Classify → ask for basic info if not enough
    3. Query Understanding → chitchat or needs retrieval?
    4. Coarse retrieve → get candidate chunks
    5. Disambiguate → enough info to commit to a disease? or ask a targeted question?
    6. Fine retrieve → treatment chunks for the committed disease
    7. Diagnose → answer with full context
    """
    query = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    traced_client = wrap_openai(client)

    # ── 1. EMERGENCY ─────────────────────────────────────────
    if _is_emergency(query):
        yield _thinking("🚨 Phát hiện tình huống khẩn cấp — đang xử lý...")
        context = _retrieve(retriever, query)
        system = _SYSTEM + (f"\n\nTài liệu tham khảo:\n{context}" if context else "")
        resp = await traced_client.chat.completions.create(
            model=diagnosis_model,
            messages=[{"role": "system", "content": system}] + _trim(messages),
            stream=True,
            langsmith_extra={"name": "vet-emergency"},
        )
        async for chunk in resp:
            if token := chunk.choices[0].delta.content:
                yield token
        return

    # ── 2. CLASSIFY → ask for basic info if missing ─────────
    yield _thinking("🔍 Đang phân tích câu hỏi...")
    followup_count = _count_followup_turns(messages)
    if followup_count < _MAX_FOLLOWUP:
        classify_result = await _classify(messages, traced_client, routing_model)
        state = classify_result.get("state", "ready")
        if state in ("vague", "has_symptoms"):
            if q := classify_result.get("followup_question", ""):
                yield q
                return

    # ── 3. QUERY UNDERSTANDING ───────────────────────────────
    yield _thinking("💬 Đang hiểu yêu cầu...")
    qu_result = await _query_understanding(messages, traced_client, routing_model)

    if not qu_result.get("needs_retrieval", True):
        if chitchat := qu_result.get("chitchat_response", ""):
            yield chitchat
            return

    # ── 4. COARSE RETRIEVE ───────────────────────────────────
    yield _thinking("📚 Đang tìm kiếm tài liệu liên quan...")
    queries = qu_result.get("queries", [])
    symptom_queries = [q["query"] for q in queries if q.get("purpose") == "symptom"]
    symptom_query = symptom_queries[0] if symptom_queries else " ".join(
        msg["content"] for msg in messages
        if msg["role"] == "user" and len(msg["content"]) > 15
    )[-400:]

    logger.info("retrieve_coarse query=%s", symptom_query[:80])
    coarse_points = _retrieve_points(retriever, symptom_query)

    if not coarse_points:
        yield "Mình chưa tìm được thông tin phù hợp. Bạn có thể mô tả thêm triệu chứng cụ thể không?"
        return

    # ── 5. DISAMBIGUATE → commit to a disease or ask more ──
    yield _thinking("🩺 Đang phân tích triệu chứng...")
    coarse_context = retriever.format_points(coarse_points)
    disambig_count = _count_disambiguate_turns(messages)
    disambig = await _disambiguate(messages, coarse_context, traced_client, routing_model)

    logger.info("disambiguate confident=%s disease=%s", disambig.get("confident"), disambig.get("disease"))

    if not disambig.get("confident") and disambig_count < _MAX_DISAMBIGUATE:
        if q := disambig.get("question", ""):
            yield q
            return

    # Commit to a disease: use disambiguate result, fallback to highest-score chunk
    top_disease = disambig.get("disease") or coarse_points[0].payload.get("disease_name", "")
    logger.info("coarse top_disease=%s", top_disease)

    # ── 6. FINE RETRIEVE → treatment chunks ──────────────────
    yield _thinking(f"💊 Đang tìm phác đồ điều trị{f': {top_disease}' if top_disease else ''}...")
    fine_points = _retrieve_treatment(retriever, top_disease) if top_disease else []
    logger.info("fine_retrieve treatment_chunks=%d disease=%s", len(fine_points), top_disease)

    seen: dict[str, object] = {}
    for p in coarse_points + fine_points:
        pid = str(p.id)
        if pid not in seen or p.score > seen[pid].score:
            seen[pid] = p
    all_points = sorted(seen.values(), key=lambda p: p.score, reverse=True)

    context = retriever.format_points(all_points)
    logger.info("context diseases=%s", _extract_disease_names(context))

    if not context:
        yield "Mình chưa tìm được thông tin phù hợp. Bạn có thể mô tả thêm triệu chứng cụ thể không?"
        return

    # ── 7. DIAGNOSE ──────────────────────────────────────────
    yield _thinking("✍️ Đang soạn câu trả lời...")
    system = _SYSTEM + f"\n\nTài liệu tham khảo:\n{context}"
    resp = await traced_client.chat.completions.create(
        model=diagnosis_model,
        messages=[{"role": "system", "content": system}] + _trim(messages),
        stream=True,
        langsmith_extra={"name": "vet-diagnose"},
    )
    async for chunk in resp:
        if token := chunk.choices[0].delta.content:
            yield token


@traceable(name="vet-retrieval")
def _retrieve(retriever: VetRetriever, query: str) -> str:
    return retriever.get_context(query)


@traceable(name="vet-retrieval-multi")
def _retrieve_points(retriever: VetRetriever, *queries: str) -> list:
    return retriever.get_points(*queries)


@traceable(name="vet-retrieval-treatment")
def _retrieve_treatment(retriever: VetRetriever, disease_name: str) -> list:
    return retriever.get_treatment_points(disease_name)


def _extract_disease_names(context: str) -> list[str]:
    import re
    return list(dict.fromkeys(
        m.strip() for m in re.findall(r'\[Bệnh: ([^|\]]+)', context)
    ))