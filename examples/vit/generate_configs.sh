#!/bin/bash

TEMPLATE="inat-template.sh"
OUT_DIR="./configs"

mkdir -p "$OUT_DIR"

IMG_SIZES=(224 384 768 1536)

ROPE_TYPES=("axial") # "mixed_axis" "hilbert")
ROPE_BASES=("learned") #2 4 8 16 32 64 128 256)
ROPE_FRACTIONS=(0.125 0.25 0.5 0.75 1.0)

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

      for FRACTION in "${ROPE_FRACTIONS[@]}"; do

        if [ "$FRACTION" = "1.0" ] || [ "$FRACTION" = "1" ]; then
          FRACTION_SUFFIX=""
        else
          FRACTION_TAG=$(echo "$FRACTION" | tr '.' 'p')
          FRACTION_SUFFIX="-frac-${FRACTION_TAG}"
        fi

        OUT_FILE="$TYPE_DIR/inat-${TYPE}-${BASE_TAG}${FRACTION_SUFFIX}-image-${IMG}-10epochs.sh"

        sed \
          -e "s/{{IMG_SIZE}}/${IMG}/g" \
          -e "s/{{ROPE_TYPE}}/${TYPE}/g" \
          -e "s/{{ROPE_BASE_TAG}}/${BASE_TAG}/g" \
          -e "s/{{ROPE_FRACTION}}/${FRACTION}/g" \
          -e "s/{{MICRO_BATCH}}/${MICRO_BATCH}/g" \
          -e "s/{{GLOBAL_BATCH}}/${GLOBAL_BATCH}/g" \
          -e "s|{{ROPE_BASE_LINE}}|${ROPE_BASE_LINE}|g" \
          "$TEMPLATE" > "$OUT_FILE"

        chmod +x "$OUT_FILE"

        echo "Generated $OUT_FILE"

      done
    done
  done
done
