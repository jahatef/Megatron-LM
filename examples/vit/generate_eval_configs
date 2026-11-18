#!/bin/bash
# Generate one eval-only config per (10-epoch checkpoint dir) x (eval image size).
# Usage: ./make-eval-configs.sh [CKPT_SEARCH_DIR]   (default: current dir)

TEMPLATE="inat-eval-template.sh"
OUT_DIR="./eval-configs"
SEARCH_DIR="${1:-.}"

EVAL_IMG_SIZES=(224 384 768 1536)

# Same number of eval samples at every resolution: EVAL_ITERS * GLOBAL_BATCH
GLOBAL_BATCH=512
EVAL_ITERS=10

# Only evaluate this checkpoint (10 epochs x 79 iters). Runs without it are skipped.
CKPT_ITER=790

declare -A MICRO_BATCH_MAP=(
  [128]=256
  [224]=128
  [384]=32
  [768]=8
  [1536]=2
)

mkdir -p "$OUT_DIR"
LIST_FILE="$OUT_DIR/all_eval_configs.txt"
: > "$LIST_FILE"
: > "$OUT_DIR/skipped_checkpoints.txt"

count=0
skipped=0

for CKPT_DIR in "$SEARCH_DIR"/checkpoints-*-10epochs; do
  [ -d "$CKPT_DIR" ] || continue
  NAME=$(basename "$CKPT_DIR")

  # checkpoints-<type>-<base>-<train_img>-<fraction>-10epochs
  if [[ ! "$NAME" =~ ^checkpoints-([a-z_]+)-([^-]+)-([0-9]+)-([0-9.]+)-10epochs$ ]]; then
    echo "WARN: can't parse '$NAME', skipping"
    skipped=$((skipped + 1))
    continue
  fi
  TYPE="${BASH_REMATCH[1]}"
  BASE="${BASH_REMATCH[2]}"
  TRAIN_IMG="${BASH_REMATCH[3]}"
  FRACTION="${BASH_REMATCH[4]}"

  CKPT_FILE=$(printf "%s/iter_%07d/mp_rank_00/model_optim_rng.pt" "$CKPT_DIR" "$CKPT_ITER")
  if [ ! -f "$CKPT_FILE" ]; then
    LATEST=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "none")
    echo "WARN: $NAME has no iter $CKPT_ITER checkpoint (latest=$LATEST), skipping"
    echo "$NAME latest=$LATEST" >> "$OUT_DIR/skipped_checkpoints.txt"
    skipped=$((skipped + 1))
    continue
  fi

  LOAD_DIR=$(realpath "$CKPT_DIR")

  if [ "$BASE" = "learned" ]; then
    ROPE_BASE_LINE="# learned rotary base"
  else
    ROPE_BASE_LINE="--vit-rotary-base $BASE"
  fi

  FRACTION_TAG="frac-$(echo "$FRACTION" | tr '.' 'p')"

  TYPE_DIR="$OUT_DIR/$TYPE"
  mkdir -p "$TYPE_DIR"

  for EVAL_IMG in "${EVAL_IMG_SIZES[@]}"; do
    MICRO_BATCH=${MICRO_BATCH_MAP[$EVAL_IMG]}
    if [ -z "$MICRO_BATCH" ]; then
      echo "Missing micro batch for EVAL_IMG=$EVAL_IMG"
      exit 1
    fi

    EXP_NAME="eval-${TYPE}-${BASE}-${FRACTION_TAG}-train${TRAIN_IMG}-eval${EVAL_IMG}"
    OUT_FILE="$TYPE_DIR/${EXP_NAME}.sh"

    sed \
      -e "s/{{TRAIN_IMG_SIZE}}/${TRAIN_IMG}/g" \
      -e "s/{{EVAL_IMG_SIZE}}/${EVAL_IMG}/g" \
      -e "s/{{ROPE_TYPE}}/${TYPE}/g" \
      -e "s/{{ROPE_BASE_TAG}}/${BASE}/g" \
      -e "s/{{ROPE_FRACTION}}/${FRACTION}/g" \
      -e "s/{{MICRO_BATCH}}/${MICRO_BATCH}/g" \
      -e "s/{{GLOBAL_BATCH}}/${GLOBAL_BATCH}/g" \
      -e "s/{{EVAL_ITERS}}/${EVAL_ITERS}/g" \
      -e "s/{{CKPT_STEP}}/${CKPT_ITER}/g" \
      -e "s|{{ROPE_BASE_LINE}}|${ROPE_BASE_LINE}|g" \
      -e "s|{{LOAD_DIR}}|${LOAD_DIR}|g" \
      -e "s|{{EXP_NAME}}|${EXP_NAME}|g" \
      "$TEMPLATE" > "$OUT_FILE"

    chmod +x "$OUT_FILE"
    echo "$OUT_FILE" >> "$LIST_FILE"
    count=$((count + 1))
  done
done

echo "Generated $count configs ($skipped checkpoint dirs skipped, see $OUT_DIR/skipped_checkpoints.txt)"
echo "List of all configs: $LIST_FILE"