# ════════════════════════════════════════════════════════════
# Stage 1: Builder — cài dependencies vào /app/.venv
# ════════════════════════════════════════════════════════════
FROM python:3.13-slim AS builder

# uv là package manager đã dùng trong dev — dùng luôn trong build
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

WORKDIR /app

# Copy requirements trước (layer cache — chỉ rebuild khi requirements đổi)
COPY requirements.txt .

# Cài vào .venv trong project, không dùng system Python
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
    --no-cache \
    -r requirements.txt

# ════════════════════════════════════════════════════════════
# Stage 2: Runtime — image nhỏ gọn, không có build tools
# ════════════════════════════════════════════════════════════
FROM python:3.13-slim AS runtime

# Security: chạy với non-root user
RUN groupadd --gid 1001 vetbot && \
    useradd --uid 1001 --gid vetbot --shell /bin/bash --create-home vetbot

WORKDIR /app

# Copy venv từ builder (không copy pip, build tools)
COPY --from=builder /app/.venv /app/.venv

# Copy source code (theo thứ tự từ ít thay đổi → hay thay đổi)
COPY vetbot/           ./vetbot/
COPY .chainlit/config.toml  ./.chainlit/config.toml
COPY chainlit.md       .
COPY config.yaml       .
COPY init_db.py        .
COPY start.sh          .
COPY app.py            .

# Đảm bảo start.sh executable
RUN chmod +x start.sh

# Runtime dir cho SQLite (dev fallback)
RUN mkdir -p data && chown -R vetbot:vetbot /app

USER vetbot

# Dùng .venv Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Railway inject $PORT — default 8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/')" || exit 1

CMD ["bash", "start.sh"]
