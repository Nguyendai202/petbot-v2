"""Core ingest logic — import từ ingest_once.py để chạy."""
import os
import re
import uuid
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pdfminer.high_level import extract_text
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct,
    VectorParams, SparseVectorParams, SparseIndexParams,
    SparseVector, NamedVector, NamedSparseVector,
    PayloadSchemaType,
)

_COLLECTION = "vet-disease"
_BATCH_SIZE = 20
_EMBEDDING_MODEL = "text-embedding-3-large"
_DENSE_SIZE = 3072

_DISEASE_HEADINGS = {
    "BỆNH CARRE",
    "BỆNH XOẮN KHUẨN",
    "PARVOVIRUS",
    "VIÊM RUỘT TRUYỀN NHIỄM DO PARVOVIRUS",
    "BỆNH VIÊM RUỘT TRUYỀN NHIỄM DO",
    "BỆNH VIÊM GAN TRUYỀN NHIỄM TRÊN CHÓ",
    "BỆNH GIẢM BẠCH CẦU MÈO",
    "BỆNH RICKETTSIA",
    "BỆNH GIUN ĐŨA",
    "BỆNH GIUN MÓC",
    "BỆNH L Ỵ DO AMIP",
    "BỆNH GIUN TIM Ở CHÓ",
    "BỆNH SÁN LÁ GAN NHỎ",
    "BỆNH CẦU TRÙNG",
    "BỆNH LỴ DO GIARDIA INTESTINALIS",
    "BỆNH LÊ DẠNG TRÙNG",
    "GHẺ NGẦM",
    "BỆNH MÒ BAO LÔNG",
    "BỆNH VIÊM PHẾ QUẢN",
    "BỆNH VIÊM PHỔI THUỲ",
    "BỆNH VIÊM MÀNG PHỔI",
    "VIÊM BÀNG QUANG",
    "BỆNH VIÊM TỬ CUNG",
    "BỆNH CO GIẬT TRƯỚC VÀ SAU KHI ĐẺ",
    "HIỆN TƯỢNG CHỬA GIẢ",
    "HỘI CHỨNG TIÊU CHẢY",
    "PHƯƠNG PHÁP CẮT ĐUÔI CHÓ",
    "THIẾN CHÓ MÈO CÁI",
    "PHƯƠNG PHÁP CẮT TAI CHÓ",
    "PHƯƠNG PHÁP MỔ ĐẺ CHÓ MÈO",
    "PHƯƠNG PHÁP CẮT NHÃN CẦU",
}

_DISEASE_NAME_MAP = {
    # Chuẩn hoá tên Parvovirus — 3 heading khác nhau trong PDF cùng 1 bệnh
    "Parvovirus": "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus",
    "Viêm Ruột Truyền Nhiễm Do Parvovirus": "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus",
    "Bệnh Viêm Ruột Truyền Nhiễm Do": "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus",
    # Chuẩn hoá tên bị lỗi spacing
    "Bệnh L Ỵ Do Amip": "Bệnh Lỵ Do Amip",
    # Thêm prefix "Bệnh" cho nhất quán
    "Viêm Bàng Quang": "Bệnh Viêm Bàng Quang",
}


# ─────────────────────────────────────────────
# SPARSE VECTOR (BM25-style keyword search)
# ─────────────────────────────────────────────

def _tokenize_vi(text: str) -> list[str]:
    """Tokenize tiếng Việt đơn giản — split theo ký tự không phải chữ."""
    text = text.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text)
    # Bỏ stopwords ngắn
    stopwords = {"và", "của", "là", "có", "không", "bé", "con", "các",
                 "cho", "với", "trong", "được", "từ", "theo", "khi"}
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def _build_sparse_vector(text: str) -> SparseVector:
    """
    Tạo sparse vector kiểu TF (term frequency) từ text.
    Dùng hash của token làm index để không cần vocabulary cố định.
    """
    tokens = _tokenize_vi(text)
    if not tokens:
        # Trả về sparse vector rỗng hợp lệ
        return SparseVector(indices=[0], values=[0.0])

    # Đếm term frequency
    tf: dict[int, float] = {}
    for token in tokens:
        idx = abs(hash(token)) % 100_000  # Hash vào 100k dimensions
        tf[idx] = tf.get(idx, 0.0) + 1.0

    # Normalize
    total = sum(tf.values())
    indices = list(tf.keys())
    values = [v / total for v in tf.values()]

    return SparseVector(indices=indices, values=values)


# ─────────────────────────────────────────────
# HEADING & SPLIT LOGIC (giữ nguyên)
# ─────────────────────────────────────────────

def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _match_heading(line: str) -> str | None:
    normalized = _normalize(line.upper())
    if normalized in _DISEASE_HEADINGS:
        clean = re.sub(r"\s+", " ", normalized).strip()
        name = clean.title()
        return _DISEASE_NAME_MAP.get(name, name)
    return None


def split_by_disease_heading(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_disease = "Chưa xác định"
    current_lines: list[str] = []

    for line in lines:
        heading = _match_heading(line)
        if heading:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_disease, content))
            current_disease = heading
            current_lines = [line]
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_disease, content))

    return sections


def split_with_disease_context(text: str, source: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    result = []
    for disease_name, section_text in split_by_disease_heading(text):
        for chunk in splitter.split_text(section_text):
            if not chunk.strip():
                continue

            # Prepend tên bệnh vào đầu mỗi chunk
            # → sparse search luôn tìm được tên bệnh
            # dù chunk nằm ở phần triệu chứng hay điều trị
            if disease_name and disease_name != "Chưa xác định":
                page_content = f"[{disease_name}]\n{chunk}"
            else:
                page_content = chunk

            result.append({
                "page_content": page_content,
                "disease_name": disease_name,
                "source": source,
            })
    return result


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("  Extracting text from PDF...")
    text = extract_text(str(path))

    print("  Splitting with disease context...")
    chunks = split_with_disease_context(text, source=path.name)

    before = len(chunks)
    chunks = [c for c in chunks if c["disease_name"] != "Chưa xác định"]
    print(f"  {before - len(chunks)} chunks 'Chưa xác định' bị loại")
    print(f"  {len(chunks)} chunks còn lại")

    disease_counts = Counter(c["disease_name"] for c in chunks)
    print("\n  Disease distribution:")
    for disease, count in disease_counts.most_common(30):
        print(f"    {disease}: {count} chunks")
    print()

    # ── Recreate collection với HYBRID vectors ────────────────
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_KEY"),
        timeout=60,
    )

    if client.collection_exists(_COLLECTION):
        client.delete_collection(_COLLECTION)
        print(f"  Deleted old collection: {_COLLECTION}")

    # Tạo collection với cả dense và sparse vectors
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=_DENSE_SIZE,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    print(f"  Created hybrid collection: {_COLLECTION}")

    # Tạo payload index cho disease_name để dùng filter trong coarse-to-fine
    client.create_payload_index(
        collection_name=_COLLECTION,
        field_name="disease_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("  Created payload index: disease_name")

    # ── Embed + build sparse + upload ────────────────────────
    openai_client = OpenAI()
    total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i: i + _BATCH_SIZE]
        texts = [c["page_content"] for c in batch]

        # Dense embedding từ OpenAI
        resp = openai_client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=texts,
        )
        dense_vectors = [e.embedding for e in resp.data]

        # Sparse vector từ TF tokenization
        sparse_vectors = [_build_sparse_vector(t) for t in texts]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
                payload=chunk,
            )
            for chunk, dense_vec, sparse_vec
            in zip(batch, dense_vectors, sparse_vectors)
        ]

        for attempt in range(3):
            try:
                client.upsert(collection_name=_COLLECTION, points=points)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Batch {i // _BATCH_SIZE + 1} retry {attempt + 1}/3 ({e})")

        print(f"  Batch {i // _BATCH_SIZE + 1}/{total_batches} uploaded")

    print(f"\n  Done! {len(chunks)} chunks ingested with hybrid vectors.")
    return {"chunks_created": len(chunks), "processed": 1}