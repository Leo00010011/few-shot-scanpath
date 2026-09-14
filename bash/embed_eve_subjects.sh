#!/bin/bash
#SBATCH --job-name=eve_senet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eve_senet_out_%j.log
#
# F3 / Stage C -- EVE subject embeddings from the released SE-Net checkpoint.
#
# Run it as `bash bash/embed_eve_subjects.sh`, NEVER `source` it: the -euo pipefail
# below would then apply to the calling shell, and one failed preflight would exit the
# login shell and drop the salloc allocation (TechStack section 1 command conventions).
#
#   salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00
#   bash bash/embed_eve_subjects.sh              # SEED=0
#   SEED=1 bash bash/embed_eve_subjects.sh       # a second draw from the same pools
#
# `sbatch bash/embed_eve_subjects.sh --export=ALL,SEED=1` works unchanged; the #SBATCH
# block is an inert comment when the script is executed directly.
#
# Every tunable is overridable from the environment. No absolute path appears here or
# in any .py (working convention 4).

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

# Per-seed directory: a second seed must not overwrite the first. This is the lesson
# bash/test_osie.sh learned when three seeds landed on one filename (TechStack 3.5a).
SEED_DIR="$OUT_DIR/seed$SEED"
EMB="$SEED_DIR/eve_fewshot_user_embedding_${NUM_FEWSHOT}_seed${SEED}.pt"

echo "== F3 Stage C =="
echo "env=$SENET_ENV seed=$SEED num_fewshot=$NUM_FEWSHOT device=$DEVICE"
echo "bridge=$BRIDGE_DIR out=$SEED_DIR"
echo "node=${SLURM_NODELIST:-$(hostname)}"

# `set -u` must be lifted across activation: this env's
# etc/conda/activate.d/libblas_mkl_activate.sh reads MKL_INTERFACE_LAYER before
# assigning it, so with nounset on, conda activate aborts the whole script with
# "MKL_INTERFACE_LAYER: unbound variable" and no F3 code ever runs. It is a hook in
# the env, not a defect in ours -- and -e/-o pipefail stay on throughout, so a genuine
# activation failure still stops the run.
# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SENET_ENV"
set -u

mkdir -p "$SEED_DIR" logs

# FR1.4 / FR11.9 -- the whole GPU run is gated on the preflight's exit code, and the
# same resolution code writes versions.txt (FR1.5), so the record cannot disagree.
python tools/eve_senet/check_env.py --versions "$SEED_DIR/versions.txt" \
    > "$SEED_DIR/check_env.json"

# The backbone init file. SE-Net/src/models.py loads data/resnet50.yaml's
# MODEL.WEIGHTS unconditionally and STRICTLY, and that pickle ships with neither this
# repo nor the checkpoint bundle -- so without this, model construction dies with a
# bare FileNotFoundError several frames inside SE-Net. It is initialisation only:
# load_model() overwrites every backbone tensor from the same checkpoint immediately
# afterwards, which is why deriving it from that checkpoint is sound. Generated once,
# never overwritten (pass --force deliberately if you later obtain the real pickle).
BACKBONE_INIT="${BACKBONE_INIT:-data/M2F_R50.pkl}"
if [ ! -f "$BACKBONE_INIT" ]; then
    echo "-- $BACKBONE_INIT absent; deriving it from $CKPT"
    python tools/eve_senet/make_backbone_init.py         --checkpoint "$CKPT" --out "$BACKBONE_INIT"         > "$SEED_DIR/backbone_init.json"
fi

# FR8 -- stdout is teed because the report summary is printed, not logged.
python tools/eve_senet/embed.py \
    --fixations   "$BRIDGE_DIR/fixations.json" \
    --subject-map "$BRIDGE_DIR/subject_id_map.json" \
    --report      "$BRIDGE_DIR/bridge_report.json" \
    --image-dir   "$BRIDGE_DIR/stimuli" \
    --checkpoint  "$CKPT" \
    --config      "$CONFIG" \
    --out-dir     "$SEED_DIR" \
    --data-root   "$DATA_ROOT" \
    --num-fewshot "$NUM_FEWSHOT" \
    --seed        "$SEED" \
    --device      "$DEVICE" \
    2>&1 | tee "$SEED_DIR/stdout.txt"

# FR9.4 -- an independent structural check, gating on its own exit code.
python tools/eve_senet/verify_embedding.py \
    --embedding   "$EMB" \
    --subject-map "$BRIDGE_DIR/subject_id_map.json" \
    --report      "$SEED_DIR/senet_report.json" \
    > "$SEED_DIR/verify.json"

# FR10.5 / working convention 6 -- our imports drop .pyc into SE-Net/. The 89 TRACKED
# __pycache__ entries stay tracked and untouched; only what this run created goes.
# Deleting by mtime would take the tracked ones too, which is why git decides here.
git ls-files --others --ignored --exclude-standard -- 'SE-Net/*.pyc' \
    | tr '\n' '\0' | xargs -0 -r rm -f
# and if importing rewrote a tracked .pyc, put it back rather than leaving SE-Net dirty
TRACKED_PYC="$(git diff --name-only -- 'SE-Net/*.pyc' || true)"
if [ -n "$TRACKED_PYC" ]; then
    echo "$TRACKED_PYC" | tr '\n' '\0' | xargs -0 -r git checkout --
fi

echo "== F3 done: $EMB =="
