#!/bin/bash
# ════════════════════════════════════════════════════════════
# VetBot startup script
# Chạy trong Docker container trên Railway
# ════════════════════════════════════════════════════════════
set -euo pipefail

echo "=== VetBot startup ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo "Python: $(python --version)"
echo "Port: ${PORT:-8000}"

# ── Database init ──────────────────────────────────────────
if [[ "${DATABASE_URL:-}" == postgres* ]]; then
    echo ">>> DB: PostgreSQL (Railway) — schema managed by Chainlit"
else
    echo ">>> DB: SQLite (dev fallback)"
    mkdir -p data
    python init_db.py
fi

# ── Graceful shutdown handler ──────────────────────────────
cleanup() {
    echo ">>> SIGTERM received — shutting down gracefully..."
    kill -TERM "$child" 2>/dev/null
    wait "$child"
    echo ">>> Shutdown complete"
}
trap cleanup SIGTERM SIGINT

# ── Start Chainlit ─────────────────────────────────────────
echo ">>> Starting Chainlit..."
chainlit run app.py \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --headless &

child=$!
wait "$child"
