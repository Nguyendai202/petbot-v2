import os
import re

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch, FusionQuery, Fusion,
    Filter, FieldCondition, MatchValue,
)

_COLLECTION = "vet-disease"
_EMBEDDING_MODEL = "text-embedding-3-large"


def _tokenize_vi(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text)
    stopwords = {"và", "của", "là", "có", "không", "bé", "con", "các",
                 "cho", "với", "trong", "được", "từ", "theo", "khi"}
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def _build_sparse_vector(text: str) -> SparseVector:
    tokens = _tokenize_vi(text)
    if not tokens:
        return SparseVector(indices=[0], values=[0.0])
    tf: dict[int, float] = {}
    for token in tokens:
        idx = abs(hash(token)) % 100_000
        tf[idx] = tf.get(idx, 0.0) + 1.0
    total = sum(tf.values())
    return SparseVector(
        indices=list(tf.keys()),
        values=[v / total for v in tf.values()],
    )




async def _expand_query(query: str, client) -> str:
    """
    Dùng LLM expand query với từ đồng nghĩa y khoa.
    VD: "uống nhiều nước" → thêm "khát nước, mất nước"
        "nôn 30 phút/lần" → thêm "nôn liên tục, tần suất nôn cao"
    """
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Bạn là chuyên gia thú y. 
Mở rộng câu hỏi sau bằng cách thêm các từ đồng nghĩa y khoa tiếng Việt.
Chỉ trả về câu mở rộng, không giải thích.
Giữ ngắn gọn, tối đa 2x độ dài gốc.

Ví dụ:
Input: "chó uống nhiều nước, nôn 30 phút/lần"
Output: "chó uống nhiều nước khát nước mất nước, nôn liên tục tần suất cao 30 phút một lần parvovirus viêm ruột"

Input: "mắt đục có màng trắng"  
Output: "mắt đục có màng trắng viêm giác mạc viêm gan truyền nhiễm"
"""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            stream=False,
        )
        expanded = resp.choices[0].message.content.strip()
        print(f"[QUERY EXPANSION] {query[:50]} → {expanded[:80]}")
        return expanded
    except Exception:
        return query

def _format_points(points: list) -> str:
    chunks = []
    for r in points:
        content = r.payload.get("page_content", "")
        if not content:
            continue
        disease = r.payload.get("disease_name", "")
        score = round(r.score, 3)
        header = f"[Bệnh: {disease} | Độ liên quan: {score}]" \
            if disease else f"[Độ liên quan: {score}]"
        chunks.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(chunks)


class VetRetriever:
    def __init__(self, top_k: int = 5):
        self._top_k = top_k
        self._client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_KEY"),
        )
        self._openai = OpenAI()

    def _query(self, search_query: str) -> list:
        resp = self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=search_query,
        )
        dense_vector = resp.data[0].embedding
        sparse_vector = _build_sparse_vector(search_query)

        return self._client.query_points(
            collection_name=_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=self._top_k * 3),
                Prefetch(query=sparse_vector, using="sparse", limit=self._top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self._top_k,
            with_payload=True,
        ).points

    def get_points(self, *queries: str) -> list:
        """Retrieve, dedup theo point ID, sort theo score giảm dần."""
        seen: dict[str, object] = {}
        for query in queries:
            for point in self._query(query):
                pid = str(point.id)
                if pid not in seen or point.score > seen[pid].score:
                    seen[pid] = point
        return sorted(seen.values(), key=lambda p: p.score, reverse=True)

    def get_treatment_points(self, disease_name: str) -> list:
        """Vòng 2 coarse-to-fine: search điều trị.
        - Pass 1: filtered theo disease_name (precision)
        - Pass 2: unfiltered với query điều trị (bắt cross-reference như Parvo→Carre)
        Dedup + sort theo score.
        """
        treatment_query = f"{disease_name} điều trị thuốc liều lượng phác đồ"
        resp = self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=treatment_query,
        )
        dense_vector = resp.data[0].embedding
        sparse_vector = _build_sparse_vector(treatment_query)

        disease_filter = Filter(
            must=[FieldCondition(
                key="disease_name",
                match=MatchValue(value=disease_name),
            )]
        )

        # Pass 1: filtered — lấy chunk điều trị của đúng bệnh
        filtered = self._client.query_points(
            collection_name=_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector, using="dense",
                         limit=self._top_k * 2, filter=disease_filter),
                Prefetch(query=sparse_vector, using="sparse",
                         limit=self._top_k * 2, filter=disease_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self._top_k,
            with_payload=True,
        ).points

        # Pass 2: generic treatment query — KHÔNG chứa tên bệnh cụ thể
        # để embedding không bị kéo về bệnh đó, bắt được cross-reference
        # VD: "Parvo điều trị → Carre" → query generic sẽ match Carre treatment chunks
        generic_query = "điều trị thuốc chống nôn bù nước huyết thanh phác đồ liều lượng"
        resp2 = self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=generic_query,
        )
        dense2 = resp2.data[0].embedding
        sparse2 = _build_sparse_vector(generic_query)

        unfiltered = self._client.query_points(
            collection_name=_COLLECTION,
            prefetch=[
                Prefetch(query=dense2, using="dense", limit=self._top_k * 2),
                Prefetch(query=sparse2, using="sparse", limit=self._top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self._top_k,
            with_payload=True,
        ).points

        # Dedup + sort
        seen: dict[str, object] = {}
        for p in filtered + unfiltered:
            pid = str(p.id)
            if pid not in seen or p.score > seen[pid].score:
                seen[pid] = p
        return sorted(seen.values(), key=lambda p: p.score, reverse=True)

    def get_context(self, query: str, expanded_query: str | None = None) -> str:
        search_query = expanded_query or query
        points = self.get_points(search_query)
        return _format_points(points)

    @staticmethod
    def format_points(points: list) -> str:
        return _format_points(points)

    def debug_payload(self, query: str) -> None:
        """Debug: xem cả dense và sparse retrieve được gì."""
        resp = self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=query,
        )
        dense_vector = resp.data[0].embedding
        sparse_vector = _build_sparse_vector(query)

        print("=== HYBRID RETRIEVE DEBUG ===")
        print(f"Query: {query[:80]}...")
        print(f"Sparse tokens: {_tokenize_vi(query)[:10]}")

        results = self._client.query_points(
            collection_name=_COLLECTION,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=5,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=5,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=5,
            with_payload=True,
        ).points

        for i, r in enumerate(results):
            print(f"\n#{i+1} score={r.score:.3f} | "
                  f"disease={r.payload.get('disease_name', 'N/A')}")
            print(f"     {r.payload.get('page_content', '')[:100]}...")
        print("=============================")