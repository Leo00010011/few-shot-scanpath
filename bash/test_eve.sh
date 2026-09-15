#!/bin/bash
#SBATCH --job-name=eve_eval
#SBATCH --output=logs/eve_eval_out_%j.log
#SBATCH --error=logs/eve_eval_err_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#
# F5 / Stage D + E -- EVE query-set evaluation with the released ISP checkpoint.
# Spec: spec/2026-09-15-eve-eval-branch/
#
# PRIMARY PATH -- interactive, under salloc, three seeds:
#
#   salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=08:00:00
#   bash bash/test_eve.sh                      # SEED=0
#   SEED=1 bash bash/test_eve.sh
#   SEED=2 bash bash/test_eve.sh
#
#   *** `bash script`, never `source script`. ***
#   `set -euo pipefail` below is scoped to the script when it is EXECUTED; under
#   `source` it applies to your shell, and one failed precondition exits the login
#   shell and takes the allocation with it.
#
# `sbatch bash/test_eve.sh` (with --export=ALL,SEED=1) still works unchanged -- the
# #SBATCH block is an inert comment when the script is executed directly.
#
# NO `module load CUDA` and NO senet env: nothing here is compiled, and F3's embedding
# is consumed as a FILE (TechStack section 1.0). Stages B, D and E all run in
# `scanpath`; detectron2 and MSDeformAttn are Stage C's alone.
#
# THE THREE THINGS THIS SCRIPT EXISTS TO MAKE IMPOSSIBLE, all of which otherwise
# produce a plausible, publishable, wrong number:
#   1. A truncated split. 354 scored images at --batch 1 is 354 batches; upstream's
#      `if i_batch > 100: break` would stop at 101 and report 29% of the test set.
#      --max_batches -1 is passed explicitly and lands in the logged arg namespace.
#   2. A permuted cohort. --fewshot_subject is GENERATED ascending and asserted before
#      it is passed; EVE_evaluation asserts again that no subject id actually moved.
#   3. A mis-scaled coordinate space. --origin_width/--origin_height are passed and
#      test.py forwards them into origin_size explicitly (the OSIE default (600, 800)
#      would squash every coordinate by 2.4x / 1.8x and every metric would still
#      return a number).
#
# Every tunable is overridable from the environment; no absolute path appears in any
# .py (working convention 4).

set -euo pipefail

echo "Starting EVE eval at: $(date)"
echo "Running on node: ${SLURM_NODELIST:-$(hostname)}"

# ---- tunables --------------------------------------------------------------
HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
ISP_ENV="${ISP_ENV:-scanpath}"
BRIDGE_DIR="${BRIDGE_DIR:-$PROJECT_DIR/data/eve_bridge}"
SENET_DIR="${SENET_DIR:-$PROJECT_DIR/data/eve_senet/seed0}"
FEATURE_DIR="${FEATURE_DIR:-$PROJECT_DIR/data/eve_features}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$PROJECT_DIR/weights/OSIE-20260904T121550Z-1-001/OSIE}"
EVE_WORK="${EVE_WORK:-$PROJECT_DIR/work/eve}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-/tmp/${USER:-eveeval}}"
STAGE_FEATURES="${STAGE_FEATURES:-1}"
SEED="${SEED:-0}"
SUBJECT_NUM="${SUBJECT_NUM:-3}"
NUM_FEWSHOT="${NUM_FEWSHOT:-10}"
N_SUBJECTS="${N_SUBJECTS:-38}"
MAX_BATCHES="${MAX_BATCHES:--1}"
MOUNT_IMAGE="${MOUNT_IMAGE:-1}"
FAST_PREFLIGHT="${FAST_PREFLIGHT:-0}"
USER_EMB="${USER_EMB:-$SENET_DIR/eve_fewshot_user_embedding_10_seed0.pt}"
# ---- end tunables ----------------------------------------------------------

BRANCH_DIR="$PROJECT_DIR/ISP/EVE/GazeformerISP"
EVAL_DIR="src/assets/EVE-eval"

mkdir -p "$EVE_WORK" "$PROJECT_DIR/logs"

echo "== F5 Stage D+E =="
echo "env=$ISP_ENV seed=$SEED subject_num=$SUBJECT_NUM num_fewshot=$NUM_FEWSHOT"
echo "bridge=$BRIDGE_DIR senet=$SENET_DIR features=$FEATURE_DIR"
# FR12.7 -- the runtime shape, printed here and recorded in preflight.json.
echo "expected: 354 batches at --batch 1, 1062 cells, 3 subjects/image, 38 participants"

# ---------------------------------------------------------------------------
# The beegfs-side preconditions first, before anything expensive. These are the ones
# that fail on a fresh cluster checkout: ALL of data/ is git-ignored, so nothing under
# it arrives by `git pull` and F2/F3/F4's artefacts are shipped by hand (F3's lesson).
# ---------------------------------------------------------------------------
for path in "$BRIDGE_DIR/fixations.json" "$BRIDGE_DIR/gt_heatmaps.h5" \
            "$BRIDGE_DIR/subject_id_map.json" "$BRIDGE_DIR/bridge_report.json" \
            "$SENET_DIR/senet_report.json" "$USER_EMB" \
            "$FEATURE_DIR/feature_report.json" "$FEATURE_DIR/embeddings.npy" \
            "$FEATURE_DIR/image_features"; do
    if [ ! -e "$path" ]; then
        echo "FATAL: $path does not exist." >&2
        echo "data/ is git-ignored, so F2/F3/F4's artefacts must be shipped by hand." >&2
        exit 1
    fi
done

CKPT="$WEIGHTS_DIR/checkpoint_best.pth"
[ -f "$CKPT" ] || { echo "FATAL: missing checkpoint $CKPT" >&2; exit 1; }
mkdir -p "$BRANCH_DIR/$EVAL_DIR/checkpoints"
ln -sf "$CKPT" "$BRANCH_DIR/$EVAL_DIR/checkpoints/checkpoint_best.pth"

# ---------------------------------------------------------------------------
# The env image. Non-fatal on purpose, exactly as bash/test_osie.sh has it: under the
# interactive salloc path this script runs several times in ONE allocation (three
# seeds, plus re-runs), the image is already mounted from the first invocation, and a
# second mount_image.py exits non-zero -- which `set -e` would turn into an abort
# before a single precondition ran. `conda activate` below is the real check.
# ---------------------------------------------------------------------------
if [ "$MOUNT_IMAGE" = "1" ]; then
    echo "-- mounting env image (non-fatal; already-mounted is expected on a re-run)"
    ( cd "$HOME_DIR" && sudo mount_image.py my_env.ext4 --rw ) || \
        echo "   mount_image.py returned $? -- continuing; conda activate is the real check"
else
    echo "-- MOUNT_IMAGE=0, skipping mount"
fi

# ---------------------------------------------------------------------------
# FR12.6 -- stage the feature cache to node-local scratch. 1062 tensors of 6.29 MB,
# each read exactly once: the same random-small-read workload F4 staged the bundle for,
# and the reason EyeNet-Pipeline/whole_train.sh gives in as many words ("to keep
# training I/O off the network filesystem"). This does NOT contradict TechStack section
# 1.3's "/tmp is node-local" warning: that is about anything which must PERSIST.
# Everything WRITTEN here still goes to beegfs.
# ---------------------------------------------------------------------------
FEATURE_SRC="$FEATURE_DIR"
if [ "$STAGE_FEATURES" = "1" ]; then
    STAGED="$LOCAL_SCRATCH/eve_features"
    mkdir -p "$STAGED"
    echo "-- staging image_features/ to $STAGED (the beegfs copy is only read)"
    rsync -a --info=progress2 "$FEATURE_DIR/image_features" "$STAGED/"
    FEATURE_SRC="$STAGED"
else
    echo "-- STAGE_FEATURES=0, reading features straight from beegfs"
fi

# ---------------------------------------------------------------------------
# Activation. conda.sh by EXPLICIT path, never $(conda info --base): before any env is
# active conda need not be on PATH at all, and the substitution would fail first.
#
# `set -u` is lifted ONLY across conda activate and restored at once -- an env whose
# etc/conda/activate.d hook reads a variable before assigning it (senet's
# libblas_mkl_activate.sh does) would otherwise abort the whole script before a line of
# our code ran (TechStack section 1.3). `scanpath` carries no such hook, but ISP_ENV is
# a tunable, so the guard stays. -e and -o pipefail are never lifted.
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
# shellcheck disable=SC1091
set +u
source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$ISP_ENV"
set -u

# test.py imports eve_eval, eve_prep and eve_bridge from tools/, and its own branch
# modules from src/.
export PYTHONPATH="$PROJECT_DIR/tools:$BRANCH_DIR/src:${PYTHONPATH:-}"

RESULT_LOG="result/$(basename "$EVAL_DIR")/log"
SEED_DIR="$BRANCH_DIR/$RESULT_LOG/seed$SEED"
mkdir -p "$SEED_DIR"

# ---- FR1.2: the resolved stack, into the log AND into a kept artefact -------
# D5, and TechStack section 1.1's lesson from F1: a run several major versions off the
# pin list reproduced the published row, and "consistent with the published numbers" is
# a coarser test than bitwise equality. A future discrepancy can only be attributed if
# each run wrote its stack down.
python - <<'PY' | tee "$EVE_WORK/versions.txt"
import importlib, sys
print("python", sys.version.replace("\n", " "))
for m in ("torch", "torchvision", "numpy", "scipy", "skimage", "cv2",
          "multimatch_gaze", "h5py", "pandas"):
    try:
        print(m, getattr(importlib.import_module(m), "__version__", "<none>"))
    except Exception as e:
        print(m, "MISSING", e)
try:
    import torch
    print("cuda", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "<no cuda device>")
except Exception as e:
    print("gpu MISSING", e)
PY

# ---- FR1.2: multimatch-gaze is THE metric-critical pin ---------------------
# `python -m pip`, not `py -m pip`: the py launcher is Windows-only. --no-deps, because
# ISP_ENV may be shared with other work and an unconditional install lets pip resolve
# multimatch-gaze's numpy/scipy/pandas requirements and move versions that work
# depends on. The verification below is what actually enforces the pin.
if ! python -c "import multimatch_gaze,sys; sys.exit(0 if multimatch_gaze.__version__=='0.1.3' else 1)" 2>/dev/null; then
  echo "multimatch-gaze 0.1.3 not present; installing (--no-deps)"
  python -m pip install --no-deps multimatch-gaze==0.1.3
fi
python - <<'PY'
import sys
import multimatch_gaze
v = getattr(multimatch_gaze, "__version__", "<none>")
if v != "0.1.3":
    sys.stderr.write(
        "FATAL: multimatch-gaze {} != 0.1.3 (metric-critical pin, TechStack section "
        "1.1); the numbers would not be comparable to the published table\n".format(v))
    sys.exit(1)
print("multimatch-gaze 0.1.3 OK")
PY

# ---- FR12.4 (D4): the cohort argument, GENERATED and then asserted ---------
# select_fewshot_subject() remaps this list to a dense 0..N-1 range, and ONLY the
# ascending order gives the identity remap that keeps row i on participant i and keeps
# exp_key_map's (name, subject) key addressing the right trial. Any other order
# silently permutes the whole cohort and every metric still looks plausible. So it is
# generated by seq, never typed, and checked before it is passed.
FEWSHOT_SUBJECTS="$(seq 0 $((N_SUBJECTS - 1)) | tr '\n' ' ')"
python - "$FEWSHOT_SUBJECTS" "$N_SUBJECTS" <<'PY'
import sys
xs = [int(v) for v in sys.argv[1].split()]
n = int(sys.argv[2])
assert xs == list(range(n)), (
    "fewshot_subject is not ascending 0..{} (D4, FR12.4): {}".format(n - 1, xs))
print("fewshot_subject ascending 0..{} OK".format(n - 1))
PY

# ---- FR11.2 / FR12.5: the preflight is the gate ----------------------------
# Its exit code aborts the run. The full FR3.4 hash sweep reads 6.7 GB and is worth
# doing once per allocation; FAST_PREFLIGHT=1 skips only that (existence is still
# checked, and preflight.json records the omission).
PREFLIGHT_ARGS=(--bridge-dir "$BRIDGE_DIR" --senet-dir "$SENET_DIR"
                --feature-dir "$FEATURE_DIR" --weights-dir "$BRANCH_DIR/$EVAL_DIR"
                --subject-num "$SUBJECT_NUM" --num-fewshot "$NUM_FEWSHOT"
                --out "$EVE_WORK/preflight.json")
if [ "$FAST_PREFLIGHT" = "1" ]; then
    PREFLIGHT_ARGS+=(--fast)
fi
python tools/eve_eval/check_eval.py "${PREFLIGHT_ARGS[@]}"

# ---- Stage D + E -----------------------------------------------------------
# cd into the branch: every relative default in test.py is anchored to
# ISP/EVE/GazeformerISP/ (TechStack section 1).
#
# $FEWSHOT_SUBJECTS is intentionally UNQUOTED: argparse nargs='+' needs 38 separate
# words. Quoting it yields fewshot_subject: ['0 1 2 ...'] and type=int then fails.
# Do not "fix" this.
#
# --feat_dir points at the STAGED copy while --feature_report and --emb_dir point at
# beegfs: the record travels with the run, the 6.7 GB does not.
#
# stdout is TEE'd, and that is not cosmetic: the headline SM/MM/SED line is a bare
# print() and the interactive path has no SLURM log. `set -o pipefail` is what makes
# the tee safe -- without it the tee's status would mask a failed run.
cd "$BRANCH_DIR"
CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject $FEWSHOT_SUBJECTS \
  --subject_num "$SUBJECT_NUM" --num_fewshot "$NUM_FEWSHOT" \
  --seed "$SEED" --eval_repeat_num 1 --batch 1 --max_batches "$MAX_BATCHES" \
  --width 512 --height 384 --origin_width 1920 --origin_height 1080 \
  --im_h 24 --im_w 32 \
  --fix_dir "$BRIDGE_DIR/fixations.json" \
  --feat_dir "$FEATURE_SRC/image_features" \
  --emb_dir "$FEATURE_DIR/embeddings.npy" \
  --img_dir "$BRIDGE_DIR/stimuli" \
  --heatmap_dir "$BRIDGE_DIR/gt_heatmaps.h5" \
  --subject_map_path "$BRIDGE_DIR/subject_id_map.json" \
  --bridge_report "$BRIDGE_DIR/bridge_report.json" \
  --senet_report "$SENET_DIR/senet_report.json" \
  --feature_report "$FEATURE_DIR/feature_report.json" \
  --user_emb_path "$USER_EMB" \
  --evaluation_dir "$EVAL_DIR" \
  2>&1 | tee "$SEED_DIR/stdout.txt"

# ---- FR10.2 / FR10.3: preserve per-seed artefacts --------------------------
# log_test_subject_{num_fewshot}_{random_support}.txt does not vary with the seed, so
# each invocation copies its outputs before the next seed overwrites them. Six files
# per seed.
cp "$BRANCH_DIR/$RESULT_LOG"/log_test_subject_*.txt \
   "$BRANCH_DIR/$RESULT_LOG/prediction.json" \
   "$BRANCH_DIR/$RESULT_LOG/metrics.json" \
   "$SEED_DIR/"
cp "$EVE_WORK/versions.txt" "$EVE_WORK/preflight.json" "$SEED_DIR/"

# The guard aggregate_seeds.py needed after the fact for F1: a mis-targeted copy pools
# non-replicates into a plausible, wrong band.
python -c "import json,sys; m=json.load(open(sys.argv[1])); \
assert m['args']['seed']==int(sys.argv[2]), \
'seed mismatch: metrics.json says %s, directory says %s (FR10.3)'%(m['args']['seed'],sys.argv[2])" \
  "$SEED_DIR/metrics.json" "$SEED"

echo "Preserved seed $SEED artefacts under $SEED_DIR:"
ls -la "$SEED_DIR"
echo "Seeds present so far: $(ls -d "$BRANCH_DIR/$RESULT_LOG"/seed* 2>/dev/null | tr '\n' ' ')"

# ---- FR10.4: pooling is NOT done here --------------------------------------
# Run all three seeds, then F6/F7 pool them. The resulting spread is the ACROSS-SEED
# one -- spread over runs, n = 3, a noisy estimate, so quote the range alongside it.
# It is NOT cur_metrics_std, which is spread over (image, subject) cells within one run
# and stays deferred to F6. A write-up labelling one as the other reports a quantity it
# did not measure.
echo "Finished at: $(date)"
