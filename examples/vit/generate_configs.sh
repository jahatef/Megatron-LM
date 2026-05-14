#!/bin/bash

TEMPLATE="inat-template.sh"
OUT_DIR="./configs"

mkdir -p $OUT_DIR

IMG_SIZES=(224 384 768 1536)
#"polar" "mixed_polar"
ROPE_TYPES=("axial" "mixed_axis" "hilbert")
ROPE_BASES=(2 4 8 16 32 64 128 "learned")

# Batch size maps (define once, not inside loop)
declare -A MICRO_BATCH_MAP=(
  [224]=128
  [384]=32
  [768]=8
  [1536]=2
)

declare -A GLOBAL_BATCH_MAP=(
  [224]=512
  [384]=512
  [768]=512
  [1536]=512
)

for IMG in "${IMG_SIZES[@]}"; do

  MICRO_BATCH=${MICRO_BATCH_MAP[$IMG]}
  GLOBAL_BATCH=${GLOBAL_BATCH_MAP[$IMG]}

  if [ -z "$MICRO_BATCH" ] || [ -z "$GLOBAL_BATCH" ]; then
    echo "Missing batch config for IMG_SIZE=$IMG"
    exit 1
  fi

  for TYPE in "${ROPE_TYPES[@]}"; do

    # ✅ Create per-rope-type directory
    TYPE_DIR="$OUT_DIR/$TYPE"
    mkdir -p "$TYPE_DIR"

    for BASE in "${ROPE_BASES[@]}"; do

      if [ "$BASE" = "learned" ]; then
        ROPE_BASE_LINE="# learned rotary base"
        BASE_TAG="learned"
      else
        ROPE_BASE_LINE="--vit-rotary-base $BASE"
        BASE_TAG="$BASE"
      fi

      OUT_FILE="$TYPE_DIR/inat-${TYPE}-${BASE_TAG}-image-${IMG}.sh"

      sed -e "s/{{IMG_SIZE}}/${IMG}/g" \
          -e "s/{{ROPE_TYPE}}/${TYPE}/g" \
          -e "s/{{ROPE_BASE_TAG}}/${BASE_TAG}/g" \
          -e "s/{{MICRO_BATCH}}/${MICRO_BATCH}/g" \
          -e "s/{{GLOBAL_BATCH}}/${GLOBAL_BATCH}/g" \
          -e "s|{{ROPE_BASE_LINE}}|${ROPE_BASE_LINE}|g" \
          "$TEMPLATE" > "$OUT_FILE"

      chmod +x "$OUT_FILE"

      echo "Generated $OUT_FILE"

    done
  done
done