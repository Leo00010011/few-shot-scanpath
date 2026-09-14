#!/bin/bash
#SBATCH --job-name=eve_senet_pin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eve_senet_pin_%j.log
#
# F3 / validation Group 5 -- pin the dead duration channel.
#
# Re-runs the SAME seed twice more with the duration input replaced, once by raw
# milliseconds and once by an absurd 1e6, and asserts both tensors come back BITWISE
# identical to the real `bins` run. That is the evidence for FR3.4's claim that
# SE-Net/src/models.py zeroes ventral_pos after adding it in, so duration never
# reaches the network. A tolerance-based check could hide a small live contribution;
# torch.equal cannot.
#
# Run it AFTER bash/embed_eve_subjects.sh has produced the reference tensor for the
# same SEED. Run as `bash bash/pin_duration_channel.sh`, never `source` it.
#
#   salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00
#   bash bash/embed_eve_subjects.sh
#   bash bash/pin_duration_channel.sh
#
# A failure here is NOT a tool bug -- see tools/eve_senet/pin_durations.py's module
# docstring and validation Group 5. It stops F5.

set -euo pipefail

SENET_ENV="${SENET_ENV:-senet}"
SEED="${SEED:-0}"
NUM_FEWSHOT="${NUM_FEWSHOT:-10}"
BRIDGE_DIR="${BRIDGE_DIR:-data/eve_bridge}"
OUT_DIR="${OUT_DIR:-data/eve_senet}"
CKPT="${CKPT:-weights/OSIE-20260904T121550Z-1-001/OSIE/ckp_11999.pt}"
CONFIG="${CONFIG:-SE-Net/configs/eve_useremb.json}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-cuda}"

SEED_DIR="$OUT_DIR/seed$SEED"
REF="$SEED_DIR/eve_fewshot_user_embedding_${NUM_FEWSHOT}_seed${SEED}.pt"

echo "== F3 Group 5 -- duration-channel pin =="
echo "env=$SENET_ENV seed=$SEED reference=$REF"
echo "node=${SLURM_NODELIST:-$(hostname)}"

if [ ! -f "$REF" ]; then
    echo "FATAL: reference tensor $REF not found -- run bash/embed_eve_subjects.sh at" \
         "SEED=$SEED first; the pin is a comparison, not a standalone run." >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SENET_ENV"

for ARM in raw absurd; do
    ARM_DIR="$SEED_DIR/pin/$ARM"
    mkdir -p "$ARM_DIR"
    echo "-- arm: $ARM"
    python tools/eve_senet/embed.py \
        --fixations   "$BRIDGE_DIR/fixations.json" \
        --subject-map "$BRIDGE_DIR/subject_id_map.json" \
        --report      "$BRIDGE_DIR/bridge_report.json" \
        --image-dir   "$BRIDGE_DIR/stimuli" \
        --checkpoint  "$CKPT" \
        --config      "$CONFIG" \
        --out-dir     "$ARM_DIR" \
        --data-root   "$DATA_ROOT" \
        --num-fewshot "$NUM_FEWSHOT" \
        --seed        "$SEED" \
        --device      "$DEVICE" \
        --duration-arm "$ARM" \
        2>&1 | tee "$ARM_DIR/stdout.txt"
done

python tools/eve_senet/pin_durations.py \
    --reference "$REF" \
    --arm "raw=$SEED_DIR/pin/raw/eve_fewshot_user_embedding_${NUM_FEWSHOT}_seed${SEED}_raw.pt" \
    --arm "absurd=$SEED_DIR/pin/absurd/eve_fewshot_user_embedding_${NUM_FEWSHOT}_seed${SEED}_absurd.pt" \
    --senet-report "$SEED_DIR/senet_report.json" \
    --out "$SEED_DIR/duration_pin.json"

# Working convention 6 -- our imports drop .pyc into SE-Net/; leave the tree clean.
git ls-files --others --ignored --exclude-standard -- 'SE-Net/*.pyc' \
    | tr '\n' '\0' | xargs -0 -r rm -f
TRACKED_PYC="$(git diff --name-only -- 'SE-Net/*.pyc' || true)"
if [ -n "$TRACKED_PYC" ]; then
    echo "$TRACKED_PYC" | tr '\n' '\0' | xargs -0 -r git checkout --
fi

echo "== F3 Group 5 done: $SEED_DIR/duration_pin.json =="
