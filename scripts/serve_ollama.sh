#!/bin/bash
# Start one Ollama server per GPU on ports 11435+, and print the endpoint list to export.
# Idempotent: an endpoint already up is skipped. OLLAMA_VULKAN=0 keeps one GPU per server.
# NGPUS overrides the detected GPU count (e.g. to reserve GPUs for something else).
set -u
STORE="$HOME/.ollama/models"
MODEL="${MODEL:-gemma4:31b}"
NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-24}"
NUM_CTX="${OLLAMA_NUM_CTX:-24576}"
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NGPUS" -ge 1 ] || { echo "no NVIDIA GPU detected; set NGPUS explicitly" >&2; exit 1; }
mkdir -p "$STORE"

for g in $(seq 0 $((NGPUS - 1))); do
    port=$((11435 + g))
    if curl -s --max-time 2 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
        echo "port $port already up"; continue
    fi
    CUDA_VISIBLE_DEVICES=$g OLLAMA_VULKAN=0 OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$STORE" \
        OLLAMA_NUM_PARALLEL="$NUM_PARALLEL" OLLAMA_CONTEXT_LENGTH="$NUM_CTX" OLLAMA_KEEP_ALIVE=24h \
        OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
        nohup ollama serve > "/tmp/ollama_gpu$g.log" 2>&1 &
    echo "started serve on $port (GPU $g): np=$NUM_PARALLEL ctx=$NUM_CTX flash=1 kv=q8_0"
done
sleep 6

if ! OLLAMA_HOST=127.0.0.1:11435 ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
    echo "pulling $MODEL into $STORE (one-time) ..."
    OLLAMA_HOST=127.0.0.1:11435 ollama pull "$MODEL"
fi

eps=""
for g in $(seq 0 $((NGPUS - 1))); do eps="$eps http://127.0.0.1:$((11435 + g))"; done
echo "=== export this for run_study_mp.py ==="
echo "  export OLLAMA_ENDPOINTS=\"${eps# }\""
