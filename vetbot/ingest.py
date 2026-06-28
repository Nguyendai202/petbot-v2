"""Core ingest logic — imported from ingest_once.py to run."""
import logging
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

logger = logging.getLogger(__name__)

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
    # Normalize Parvovirus name — 3 different headings in the PDF for the same disease
    "Parvovirus": "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus",
    "Viêm Ruột Truyền Nhiễm Do Parvovirus": "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus",
    "Bệnh Viêm Ruột Truyền Nhiễm Do": "Bệnh Viêm Ruột Truyền Nhiễm Do Parvovirus",
    # Fix a name with broken spacing
    "Bệnh L Ỵ Do Amip": "Bệnh Lỵ Do Amip",
    # Add "Bệnh" prefix for consistency
    "Viêm Bàng Quang": "Bệnh Viêm Bàng Quang",
}


# ─────────────────────────────────────────────
# SPARSE VECTOR (BM25-style keyword search)
# ─────────────────────────────────────────────

def _tokenize_vi(text: str) -> list[str]:
    """Simple Vietnamese tokenizer — splits on non-word characters."""
    text = text.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text)
    # Drop short stopwords
    stopwords = {"và", "của", "là", "có", "không", "bé", "con", "các",
                 "cho", "với", "trong", "được", "từ", "theo", "khi"}
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def _build_sparse_vector(text: str) -> SparseVector:
    """
    Build a TF (term frequency) sparse vector from text.
    Uses a token hash as the index so no fixed vocabulary is needed.
    """
    tokens = _tokenize_vi(text)
    if not tokens:
        # Return a valid empty sparse vector
        return SparseVector(indices=[0], values=[0.0])

    # Count term frequency
    tf: dict[int, float] = {}
    for token in tokens:
        idx = abs(hash(token)) % 100_000  # Hash into 100k dimensions
        tf[idx] = tf.get(idx, 0.0) + 1.0

    # Normalize
    total = sum(tf.values())
    indices = list(tf.keys())
    values = [v / total for v in tf.values()]

    return SparseVector(indices=indices, values=values)


# ─────────────────────────────────────────────
# HEADING & SPLIT LOGIC (unchanged)
# ─────────────────────────────────────────────

_CROSS_REF_TRIGGERS = {"tương tự", "xem bệnh", "xem điều trị", "theo bệnh"}


def _detect_cross_ref(text: str, openai_client) -> str | None:
    """Use an LLM to detect treatment cross-references, e.g. 'similar to Carre disease'.
    Only called when the chunk contains a trigger keyword, to save API calls.
    """
    lower = text.lower()
    if not any(t in lower for t in _CROSS_REF_TRIGGERS):
        return None

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Đoạn văn sau có nhắc đến 'điều trị tương tự bệnh X' "
                    "hoặc 'xem điều trị bệnh X' không?\n"
                    "Nếu có: trả về đúng tên bệnh đó (chỉ tên, không giải thích).\n"
                    "Nếu không: trả về từ null.\n\n"
                    f"{text[:400]}"
                ),
            }
        ],
        max_tokens=20,
        temperature=0,
    )
    result = resp.choices[0].message.content.strip()
    if result.lower() == "null" or not result:
        return None
    name = result.strip().rstrip(".")
    if not name.lower().startswith("bệnh"):
        name = "Bệnh " + name
    return name


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

            # Prepend the disease name to every chunk
            # → sparse search can always find the disease name
            # regardless of whether the chunk is symptoms or treatment
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

    logger.info("Extracting text from PDF...")
    text = extract_text(str(path))

    logger.info("Splitting with disease context...")
    chunks = split_with_disease_context(text, source=path.name)

    before = len(chunks)
    chunks = [c for c in chunks if c["disease_name"] != "Chưa xác định"]
    logger.info("%d 'unidentified' chunks dropped, %d chunks remaining",
                before - len(chunks), len(chunks))

    disease_counts = Counter(c["disease_name"] for c in chunks)
    logger.info("Disease distribution: %s", dict(disease_counts.most_common(30)))

    # ── Recreate collection with HYBRID vectors ───────────────
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_KEY"),
        timeout=60,
    )

    if client.collection_exists(_COLLECTION):
        client.delete_collection(_COLLECTION)
        logger.info("Deleted old collection: %s", _COLLECTION)

    # Create collection with both dense and sparse vectors
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
    logger.info("Created hybrid collection: %s", _COLLECTION)

    # Create a payload index on disease_name for coarse-to-fine filtering
    client.create_payload_index(
        collection_name=_COLLECTION,
        field_name="disease_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Created payload index: disease_name")

    # ── Detect cross-references ───────────────────────────────
    openai_client = OpenAI()
    cross_ref_count = 0
    for chunk in chunks:
        ref = _detect_cross_ref(chunk["page_content"], openai_client)
        chunk["cross_ref_disease"] = ref
        if ref:
            cross_ref_count += 1
            logger.info("cross_ref disease=%s ref=%s", chunk["disease_name"], ref)
    logger.info("%d cross-references detected", cross_ref_count)

    # ── Embed + build sparse + upload ────────────────────────
    total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i: i + _BATCH_SIZE]
        texts = [c["page_content"] for c in batch]

        # Dense embedding from OpenAI
        resp = openai_client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=texts,
        )
        dense_vectors = [e.embedding for e in resp.data]

        # Sparse vector from TF tokenization
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
                logger.warning("Batch %d retry %d/3 (%s)", i // _BATCH_SIZE + 1, attempt + 1, e)

        logger.info("Batch %d/%d uploaded", i // _BATCH_SIZE + 1, total_batches)

    logger.info("Done! %d chunks ingested with hybrid vectors.", len(chunks))
    return {"chunks_created": len(chunks), "processed": 1}