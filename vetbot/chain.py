import json
from typing import AsyncGenerator

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from .retriever import VetRetriever

_SYSTEM = """\
Bạn là bác sĩ thú y đang tư vấn cho chủ nuôi chó mèo qua tin nhắn. \
Kiến thức dựa trên sách "Bệnh của Chó, Mèo" (TS. Vũ Như Quán, 2008). \
Trả lời bằng tiếng Việt, ngắn gọn như nhắn tin Zalo.

CÁCH ĐỌC TÀI LIỆU THAM KHẢO:
- Đọc toàn bộ context, xác định TÊN BỆNH — không phải triệu chứng
- Ưu tiên chunk có score cao hơn
- Nếu nhiều bệnh có thể → liệt kê theo thứ tự khả năng cao nhất
- Chunk không liên quan → bỏ qua hoàn toàn

KHI CONTEXT CÓ NHIỀU CHUNK CÙNG BỆNH:
- Tổng hợp thông tin từ tất cả chunk của bệnh đó
- Chunk triệu chứng → xác nhận chẩn đoán
- Chunk điều trị → đưa ra hướng dẫn cụ thể
- KHÔNG bỏ qua chunk điều trị dù score thấp hơn

PHÂN LOẠI MỨC ĐỘ:
🟢 Nhẹ — tự xử lý tại nhà
🟡 Theo dõi — có thể dùng thuốc OTC
🟠 Cần bác sĩ — ra thú y trong 24h
🔴 Khẩn cấp — đưa đi ngay, không chờ

FORMAT TRẢ LỜI BỆNH:
[Emoji] Chẩn đoán: [tên bệnh đầy đủ — KHÔNG phải triệu chứng]

Xử lý ngay: [hành động cụ thể người dùng làm được tại nhà]

Thuốc có thể dùng: [tên thuốc + liều lượng cụ thể, có bán tại VN]

Ra thú y ngay nếu: [dấu hiệu cụ thể cần đến gặp bác sĩ]

_(Phòng bệnh lần sau: [1 câu ngắn — vaccine/vệ sinh/dinh dưỡng])_

QUY TẮC FORMAT ĐIỀU TRỊ:
- Ưu tiên thông tin CHỮA BỆNH — đây là thứ người dùng cần ngay
- Thuốc phải cụ thể: tên thương mại, liều theo kg, cách dùng
- Phòng bệnh chỉ ghi 1 câu in nghiêng ở cuối — không mở rộng
- Nếu bệnh nặng cần thú y: vẫn hướng dẫn xử lý TẠM THỜI trong lúc di chuyển
- Với bệnh truyền nhiễm nguy hiểm: nêu rõ tỷ lệ nguy hiểm ngắn gọn
  VD: "Parvo tỷ lệ chết 40-50% nếu không điều trị kịp"

FORMAT TRẢ LỜI SẢN PHẨM/DINH DƯỠNG:
Nhận xét: [dựa trên thành phần nếu có, nếu không có data → nói thẳng]
Phù hợp với: [tình trạng nên dùng]
Không phù hợp với: [tình trạng nên tránh]
Nên kết hợp thêm: [gợi ý bổ sung]
Lưu ý: [giới hạn sử dụng]

QUY TẮC:
- KHÔNG bắt đầu bằng "Dựa trên...", "Theo tài liệu..."
- KHÔNG tóm tắt lại triệu chứng người dùng đã nói
- Luôn gọi thú cưng là "bé"
- Nếu không có data sản phẩm: nói thẳng, đừng bịa

EMERGENCY — 🔴 + sơ cứu NGAY, không hỏi thêm:

Co giật/động kinh:
  Không giữ chặt bé. Để nằm mặt phẳng, tránh đập đầu.
  Tắt đèn sáng và tiếng ồn. Không cho ăn uống gì. Đưa thú y ngay.
  Nguyên nhân có thể: Carré thể thần kinh, thiếu canxi cấp,
  ngộ độc — cần xét nghiệm xác định, không tự chẩn đoán.

Liệt đột ngột:
  Không để bé tự đi. Đặt nằm phẳng, giữ ấm. Đưa thú y ngay.

Khó thở/tím lưỡi:
  Không để nằm ngửa. Đưa thú y ngay — khẩn cấp tuyệt đối.

Nôn ra máu:
  Không cho ăn uống. Ghi lại tần suất. Đưa thú y ngay.

Bất tỉnh:
  Kiểm tra còn thở không. Giữ ấm. Đưa thú y ngay.

Không tiểu được (đặc biệt mèo đực):
  Khẩn cấp tuyệt đối — tắc niệu đạo có thể tử vong trong vài giờ.
  Đưa thú y ngay, không chờ sáng.\
"""

_CLASSIFY_SYSTEM = """\
Bạn là AI phân loại trạng thái hội thoại tư vấn thú y.
Phân tích cuộc trò chuyện và trả về JSON.

OUTPUT (JSON only):
{
  "state": "vague|has_symptoms|product_query|ready",
  "followup_question": "câu hỏi tiếp theo (empty nếu ready/product_query)"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4 TRẠNG THÁI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"vague" — Chưa có triệu chứng cụ thể
Ví dụ: "bé trông không ổn", "bé buồn", "bé lạ lạ"
→ Hỏi triệu chứng cụ thể:
  "Bé có biểu hiện gì cụ thể không — bỏ ăn, nôn,
   tiêu chảy, ho, rụng lông, hay dấu hiệu nào bạn thấy lạ?"

─────────────────────────────

"product_query" — Hỏi về sản phẩm/dinh dưỡng cụ thể
Nhận biết: có tên thức ăn cụ thể (Kucinta, Nekko, CattyMan,
           Royal Canin, Whiskas...) hoặc hỏi về thành phần
→ followup_question: "" — trả lời ngay, không hỏi triệu chứng

─────────────────────────────

"has_symptoms" — Có triệu chứng nhưng thiếu thông tin phân biệt

BƯỚC 1: Kiểm tra thông tin cơ bản — hỏi nếu thiếu:
□ Loài (chó/mèo)
□ Tuổi (tháng)
□ Tiêm phòng đầy đủ chưa

BƯỚC 2: Sau khi có thông tin cơ bản,
hỏi dấu hiệu PHÂN BIỆT theo nhóm:

TIÊU HÓA (nôn/tiêu chảy/bỏ ăn):
  ✦ Màu chất nôn (trắng bọt / vàng mật / có máu)
  ✦ Tần suất nôn (mấy lần/ngày, cách bao lâu/lần)
  ✦ Phân: màu gì, có mùi tanh khắm không, có máu không
  ✦ Bé có uống nhiều nước hơn bình thường không
  ✦ Mắt có bị đục hoặc có màng trắng không
  ✦ Trước đó có ho hoặc chảy mũi không

HÔ HẤP (ho/khó thở/chảy mũi):
  ✦ Dịch mũi màu gì (trong / vàng / xanh)
  ✦ Ho khan hay có đờm
  ✦ Bé có sốt không (tai/bụng nóng hơn bình thường)

THẦN KINH (run/yếu chân/mất thăng bằng):
  ✦ Lần đầu hay đã từng bị
  ✦ Trước đó có sốt/chảy mũi/tiêu chảy không
  ✦ Đực hay cái — có mang thai hoặc mới đẻ không

DA/LÔNG (rụng lông/ngứa/mụn):
  ✦ Rụng từng mảng hay toàn thân
  ✦ Da có đỏ/chảy dịch/mủ không
  ✦ Có ve rận gần đây không

TIẾT NIỆU/MÁU:
  ✦ Màu nước tiểu (vàng / đục / đỏ nâu)
  ✦ Có rặn tiểu không
  ✦ Đực hay cái — có ve rận không

─────────────────────────────

"ready" — Đủ thông tin để chẩn đoán
→ followup_question: ""

ĐIỀU KIỆN ready CHO NHÓM TIÊU HÓA:
✅ Loài + tuổi + tiêm phòng
✅ Màu chất nôn + tần suất nôn
✅ Đặc điểm phân (màu + mùi)
✅ Có uống nhiều nước không
✅ Mắt có đục không

VÍ DỤ READY NGAY:
• "Chó 3 tháng chưa tiêm, nôn mỗi 30 phút, phân vàng
   mùi tanh khắm, uống rất nhiều nước, lờ đờ" → ready
• "Chó 5 tháng tiêm đủ, nôn vàng, mắt đục, bỏ ăn" → ready

VÍ DỤ CHƯA READY:
• "Bé nôn và bỏ ăn" → has_symptoms
• "Chó 5 tháng tiêm đủ, nôn vàng, bỏ ăn" → has_symptoms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LƯU Ý
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Chỉ hỏi những gì CHƯA được trả lời
- Gộp tối đa 3 câu hỏi vào 1 lượt
- Thân thiện như nhắn tin Zalo\
"""

_QUERY_UNDERSTANDING_SYSTEM = """\
Bạn là AI chuyên phân tích câu hỏi tư vấn thú y để chuẩn bị retrieve tài liệu.

Phân tích toàn bộ cuộc trò chuyện và trả về JSON.

OUTPUT (JSON only):
{
  "needs_retrieval": true/false,
  "queries": [
    {
      "purpose": "symptom|treatment|product",
      "query": "câu query tối ưu để retrieve"
    }
  ],
  "chitchat_response": "câu trả lời nếu không cần retrieve (empty nếu cần retrieve)"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOGIC PHÂN TÍCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

needs_retrieval = false KHI:
- Tin nhắn thuần túy xã giao: "xin chào", "cảm ơn", "ok", "được rồi"
- Câu hỏi không liên quan thú y
→ queries: []
→ chitchat_response: câu trả lời ngắn phù hợp

needs_retrieval = true KHI:
- Có triệu chứng bệnh cụ thể
- Hỏi về sản phẩm thức ăn/dinh dưỡng
- Hỏi về điều trị/thuốc
→ chitchat_response: ""
→ Tạo queries phù hợp

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÁCH TẠO QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Luôn tạo 2 queries khi có triệu chứng bệnh:

Query 1 — purpose: "symptom"
Tập trung vào triệu chứng để tìm chunk chẩn đoán.
Chỉ lấy thông tin triệu chứng thật từ conversation.
KHÔNG thêm tên bệnh nếu chưa chắc chắn.
VD: "chó 3 tháng chưa tiêm nôn vàng 30 phút phân tanh uống nhiều nước"

Query 2 — purpose: "treatment"
Tập trung vào điều trị để tìm chunk thuốc/phác đồ.
Chỉ thêm tên bệnh nếu đã được xác nhận từ triệu chứng rõ ràng.
VD: "viêm ruột truyền nhiễm parvovirus điều trị thuốc phác đồ chống nôn"
Nếu chưa rõ bệnh, dùng triệu chứng chính + từ khóa điều trị:
VD: "chó nôn vàng tiêu chảy tanh điều trị thuốc chống nôn bù nước"
KHÔNG dùng query quá chung như "chó nôn tiêu chảy điều trị"

Khi hỏi về sản phẩm — 1 query duy nhất:
Query 1 — purpose: "product"
VD: "Kucinta pate thành phần dinh dưỡng mèo tiết niệu"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC QUAN TRỌNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Query chỉ chứa thông tin THẬT từ conversation
- KHÔNG đoán hay thêm tên bệnh vào query symptom
- Query phải ngắn gọn, súc tích — tối đa 15 từ
- Bỏ từ xã giao, giữ triệu chứng và thông tin y tế\
"""

_MAX_HISTORY_TURNS = 10
_MAX_FOLLOWUP = 4


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
    Chỉ xác định: chitchat hay cần retrieve?
    Nếu cần retrieve → dùng thẳng thông tin từ conversation làm query,
    không rewrite lại để tránh thêm giả định.
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
        print(f"[QUERY UNDERSTANDING] needs_retrieval={result.get('needs_retrieval')}")
        return result
    except Exception:
        return {"needs_retrieval": True, "chitchat_response": ""}



async def stream_answer(
    retriever: VetRetriever,
    messages: list[dict],
    client: AsyncOpenAI,
    model: str = "gpt-4o-mini",
) -> AsyncGenerator[str, None]:
    """
    Flow:
    1. Emergency check → sơ cứu ngay
    2. Classify → hỏi thêm nếu chưa đủ thông tin
    3. Query Understanding → xác định queries tối ưu
    4. Retrieve với từng query → merge context
    5. Context relevance check → chẩn đoán hoặc hỏi thêm
    """
    query = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    traced_client = wrap_openai(client)

    # ── 1. EMERGENCY ─────────────────────────────────────────
    if _is_emergency(query):
        context = _retrieve(retriever, query)
        system = _SYSTEM + (f"\n\nTài liệu tham khảo:\n{context}" if context else "")
        resp = await traced_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + _trim(messages),
            stream=True,
            langsmith_extra={"name": "vet-emergency"},
        )
        async for chunk in resp:
            if token := chunk.choices[0].delta.content:
                yield token
        return

    # ── 2. CLASSIFY → hỏi thêm nếu chưa đủ ─────────────────
    followup_count = _count_followup_turns(messages)
    if followup_count < _MAX_FOLLOWUP:
        classify_result = await _classify(messages, traced_client, model)
        state = classify_result.get("state", "ready")
        if state in ("vague", "has_symptoms"):
            if q := classify_result.get("followup_question", ""):
                yield q
                return

    # ── 3. QUERY UNDERSTANDING ───────────────────────────────
    qu_result = await _query_understanding(messages, traced_client, model)

    # Không cần retrieve → chitchat
    if not qu_result.get("needs_retrieval", True):
        chitchat = qu_result.get("chitchat_response", "")
        if chitchat:
            yield chitchat
            return

    # ── 4. COARSE-TO-FINE RETRIEVE ───────────────────────────
    symptom_query = " ".join(
        msg["content"] for msg in messages
        if msg["role"] == "user" and len(msg["content"]) > 15
    )[-600:]

    # Vòng 1 — Coarse: tìm bệnh từ triệu chứng
    print(f"[RETRIEVE:coarse] {symptom_query[:80]}")
    coarse_points = _retrieve_points(retriever, symptom_query)

    if not coarse_points:
        yield (
            "Mình chưa tìm được thông tin phù hợp. "
            "Bạn có thể mô tả thêm triệu chứng cụ thể không?"
        )
        return

    # Xác định bệnh có score cao nhất từ vòng 1
    top_disease = coarse_points[0].payload.get("disease_name", "")
    print(f"[COARSE] top disease: {top_disease}")

    # Vòng 2 — Fine: lấy chunk điều trị của đúng bệnh đó
    fine_points = _retrieve_treatment(retriever, top_disease) if top_disease else []
    print(f"[FINE] {len(fine_points)} treatment chunks for: {top_disease}")

    # Merge coarse + fine, dedup theo ID, sort theo score
    seen: dict[str, object] = {}
    for p in coarse_points + fine_points:
        pid = str(p.id)
        if pid not in seen or p.score > seen[pid].score:
            seen[pid] = p
    all_points = sorted(seen.values(), key=lambda p: p.score, reverse=True)

    context = retriever.format_points(all_points)

    print("[CONTEXT RELEVANCE INPUT]")
    print(f"Query: {symptom_query[:100]}")
    print(f"Context diseases: {_extract_disease_names(context)}")

    if not context:
        yield (
            "Mình chưa tìm được thông tin phù hợp. "
            "Bạn có thể mô tả thêm triệu chứng cụ thể không?"
        )
        return

    # ── 6. DIAGNOSE ──────────────────────────────────────────
    system = _SYSTEM + f"\n\nTài liệu tham khảo:\n{context}"
    resp = await traced_client.chat.completions.create(
        model=model,
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