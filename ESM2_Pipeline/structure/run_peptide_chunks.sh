#!/bin/bash
# ============================================================================
# Resilient peptide generation in chunks with retries
# - Splits peptide_unique.fasta into smaller chunks
# - Runs colabfold_batch per chunk with retries
# - Sets XLA memory flags to reduce OOMs
# ============================================================================

set -euo pipefail

STRUCTURE_DIR="/home3/rudrapra001/FYP-Research/ESM2_Pipeline/structure"
FASTA="$STRUCTURE_DIR/peptide_unique.fasta"
OUT_DIR="$STRUCTURE_DIR/data/generated_pdbs/peptide"
# Allow overriding chunk dir to support multiple workers safely
CHUNK_DIR="${CHUNK_DIR:-$STRUCTURE_DIR/tmp/peptide_chunks}"
CHUNK_SIZE=${CHUNK_SIZE:-500}   # override via env if desired
MAX_RETRIES=3
SLEEP_BETWEEN=15

COLABFOLD_BIN="$STRUCTURE_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch"
COLABFOLD_LIB="$STRUCTURE_DIR/localcolabfold/colabfold-conda/lib"
export LD_LIBRARY_PATH="$COLABFOLD_LIB:${LD_LIBRARY_PATH:-}"
# Reduce XLA preallocation to avoid OOMs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

mkdir -p "$OUT_DIR" "$CHUNK_DIR"
cd "$STRUCTURE_DIR"

# Only split if no chunks exist in this CHUNK_DIR
if ! ls "$CHUNK_DIR"/chunk_*.fasta >/dev/null 2>&1; then
    echo "Splitting peptides into chunks of $CHUNK_SIZE sequences into $CHUNK_DIR..."
    python - <<'PY'
import sys, os
from pathlib import Path
CHUNK_DIR = Path(os.environ.get("CHUNK_DIR", "tmp/peptide_chunks"))
FASTA = Path("peptide_unique.fasta")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

chunks = []
current = []
with FASTA.open() as f:
    for line in f:
        if line.startswith('>') and len(current) >= 2*CHUNK_SIZE:  # ~2 lines per seq
            out = CHUNK_DIR / f"chunk_{len(chunks):05d}.fasta"
            out.write_text(''.join(current))
            chunks.append(out)
            current = []
        current.append(line)
if current:
    out = CHUNK_DIR / f"chunk_{len(chunks):05d}.fasta"
    out.write_text(''.join(current))
    chunks.append(out)

print(f"Created {len(chunks)} chunk files in {CHUNK_DIR}")
PY
else
    echo "Chunks already present in $CHUNK_DIR; reusing without re-splitting."
fi

echo "Starting chunked peptide generation from $CHUNK_DIR..."
for chunk in "$CHUNK_DIR"/chunk_*.fasta; do
    [ -e "$chunk" ] || continue

    # Atomic lock to avoid two workers processing same file
    lockfile="${chunk}.lock"
    if ! ( set -o noclobber; > "$lockfile" ) 2>/dev/null; then
        # Someone else is processing this chunk
        continue
    fi

    base=$(basename "$chunk")
    echo "\n==> Processing $base"
    attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "  Attempt $attempt..."
        if "$COLABFOLD_BIN" \
            --msa-mode single_sequence \
            --num-models 1 \
            --num-recycle 1 \
            "$chunk" \
            "$OUT_DIR"; then
            echo "  ✓ Success: $base"
            break
        fi
        echo "  ⚠ Failed attempt $attempt, retrying after $SLEEP_BETWEEN s..."
        sleep $SLEEP_BETWEEN
        attempt=$((attempt+1))
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "❌ Giving up on $base after $MAX_RETRIES attempts" >&2
    else
        # Cleanup chunk to avoid reprocessing on reruns
        rm -f "$chunk"
    fi
    rm -f "$lockfile"
done

echo "\nAll chunks processed. Check outputs in $OUT_DIR"