#!/bin/bash
#SBATCH --job-name=cocofv_eval
#SBATCH --output=logs/cocofv_out_%j.log
#SBATCH --error=logs/cocofv_err_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal
#
# COCO-FreeView few-shot query-set evaluation -- Roadmap F1 baseline.
# Spec: spec/2026-09-08-cocofv-baseline-on-cluster/
#
#   sbatch bash/test_cocofv.sh                              # SEED=0
#   sbatch --export=ALL,SEED=1 bash/test_cocofv.sh
#   sbatch --export=ALL,SEED=2 bash/test_cocofv.sh
#
# Every tunable below is overridable from the environment, so a variant run needs
# no edit to the script body (FR8.4).

set -euo pipefail

echo "Starting COCO-FreeView eval at: $(date)"
echo "Running on node: ${SLURM_NODELIST:-<not under slurm>}"

# ---- tunables (FR8.4) ------------------------------------------------------
HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
FV_IMAGE_ROOT="${FV_IMAGE_ROOT:-$PROJECT_DIR/data/COCO_FV}"
FV_FIX_JSON="${FV_FIX_JSON:-$PROJECT_DIR/data/coco_fv_fixations_update_duration.json}"
FV_WORK="${FV_WORK:-$PROJECT_DIR/work/cocofv}"   # BeeGFS -- features stream from here (FR4.7)
ISP_ENV="${ISP_ENV:-isp}"
FEWSHOT_SUBJECTS="${FEWSHOT_SUBJECTS:-0 1 2}"
SUBJECT_NUM="${SUBJECT_NUM:-3}"
SEED="${SEED:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-/tmp/${USER:-cocofv}}"
# ---- end tunables ----------------------------------------------------------

BRANCH_DIR="$PROJECT_DIR/ISP/COCO_FV/GazeformerISP"
PREP="$PROJECT_DIR/tools/cocofv_prep"
FV_DATA="$FV_WORK/data/FV"

cd "$HOME_DIR"
echo "Mounting image"
sudo mount_image.py my_env.ext4 --rw

# ---- FR2.4: cheap preconditions before anything expensive ------------------
if [ ! -d "$FV_IMAGE_ROOT" ]; then
  echo "FATAL: FV_IMAGE_ROOT does not exist: $FV_IMAGE_ROOT" >&2; exit 1
fi
if [ -z "$(find "$FV_IMAGE_ROOT" -mindepth 2 -maxdepth 2 -name '*.jpg' -print -quit)" ]; then
  echo "FATAL: FV_IMAGE_ROOT has no <category>/*.jpg under it: $FV_IMAGE_ROOT" >&2; exit 1
fi
if [ ! -f "$FV_FIX_JSON" ]; then
  echo "FATAL: FV_FIX_JSON does not exist: $FV_FIX_JSON" >&2; exit 1
fi

# ---- scratch staging: SMALL inputs only ------------------------------------
# The image_features/ cache is tens of GB (one 6.3 MB .pth per stimulus) and is
# read straight from BeeGFS, NOT copied into the size-limited scratch. The
# reference train_ms.sh records a run that silently failed to rsync a 22 GB cache
# and then could not find it. Do not "optimise" this into an rsync. (FR4.7, FR8.7)
mkdir -p "$LOCAL_SCRATCH/cocofv"
cp "$FV_FIX_JSON" "$LOCAL_SCRATCH/cocofv/labels_raw.json"

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$ISP_ENV"

# ---- FR1.2: resolved versions into the SLURM log ---------------------------
python - <<'PY'
import importlib, sys
print("python", sys.version.replace("\n", " "))
for m in ("torch", "numpy", "scipy", "skimage", "multimatch_gaze", "cv2"):
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "MISSING", e)
PY

# ---- FR1.3: multimatch-gaze is metric-critical, pinned exactly --------------
py -m pip install multimatch-gaze==0.1.3
python - <<'PY'
import sys
import multimatch_gaze
v = getattr(multimatch_gaze, "__version__", "<none>")
if v != "0.1.3":
    sys.stderr.write(
        "FATAL: multimatch-gaze {} != 0.1.3 (metric-critical pin, TechStack section 1);"
        " the numbers would not be comparable to the published table\n".format(v))
    sys.exit(1)
print("multimatch-gaze 0.1.3 OK")
PY

# ---- FR3: canonical fixations.json -----------------------------------------
cd "$PROJECT_DIR"
mkdir -p "$FV_DATA"
python "$PREP/normalize_fixations.py" \
    --in "$LOCAL_SCRATCH/cocofv/labels_raw.json" \
    --out "$FV_DATA/fixations.json"

# ---- FR3.5 preflight; stdout is the counter JSON, stderr is the commentary --
python "$PREP/check_fixations.py" \
    --fix "$FV_DATA/fixations.json" \
    --images "$FV_IMAGE_ROOT" \
    --fewshot-subject $FEWSHOT_SUBJECTS \
    --split test \
    | tee "$FV_WORK/preflight_fixations.json"

# ---- FR5.4: the checked-in embeddings.npy already satisfies FR5.2 ----------
# Verified on the dev machine: {"free-viewing": (768,) float32}. Copying it means
# the run needs no hub access from the compute node at all, so FR5.3's
# HF_HUB_OFFLINE dance is unnecessary. text_data() is deliberately NOT run.
cp "$BRANCH_DIR/src/embeddings.npy" "$FV_DATA/embeddings.npy"
python - "$FV_DATA/embeddings.npy" <<'PY'
import sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True).item()
v = d["free-viewing"]
assert v.shape == (768,), "embeddings.npy free-viewing shape {} != (768,)".format(v.shape)
assert v.dtype == np.float32, "embeddings.npy free-viewing dtype {} != float32".format(v.dtype)
print("embeddings.npy OK: free-viewing", v.shape, v.dtype)
PY

# ---- Stage B, guarded (FR8.6) ----------------------------------------------
# cd first: every relative default in feature_extractor.py and test.py is anchored
# to ISP/COCO_FV/GazeformerISP/ (FR7.1). The preflight tools are called by absolute
# path so they work from either directory.
cd "$BRANCH_DIR"
if [ "$FORCE_FEATURES" = "1" ] || \
   ! python "$PREP/check_features.py" --fix "$FV_DATA/fixations.json" \
                                      --feat-dir "$FV_DATA/image_features" ; then
  echo "Extracting image features into $FV_DATA/image_features"
  python src/preprocess/feature_extractor.py \
      --dataset_path "$FV_IMAGE_ROOT" --output_path "$FV_DATA" --cuda 0
  python "$PREP/check_features.py" --fix "$FV_DATA/fixations.json" \
                                   --feat-dir "$FV_DATA/image_features"
else
  echo "Feature cache complete; skipping extraction (FORCE_FEATURES=1 to override)"
fi

# ---- Stage D + E (FR7.1, FR7.2) --------------------------------------------
# $FEWSHOT_SUBJECTS is intentionally UNQUOTED: argparse nargs='+' needs three
# separate words. Quoting it yields fewshot_subject: ['0 1 2'] and argparse's
# type=int then fails. Do not "fix" this.                            (FR8.4)
CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject $FEWSHOT_SUBJECTS \
  --subject_num "$SUBJECT_NUM" \
  --seed "$SEED" \
  --eval_repeat_num 1 \
  --fix_dir  "$FV_DATA/fixations.json" \
  --feat_dir "$FV_DATA/image_features" \
  --emb_dir  "$FV_DATA/embeddings.npy" \
  --img_dir  "$FV_IMAGE_ROOT" \
  --evaluation_dir src/assets/FV-ex-012 \
  --user_emb_path ../../../SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt

# ---- FR15.2: preserve per-seed artefacts before the next seed overwrites ----
# All three seeds write the same log_test_subject_10_0.txt (the filename tracks
# --num_fewshot/--random_support, neither of which affects a query-set score, FR9.5).
mkdir -p "result/FV-ex-012/log/seed$SEED"
cp result/FV-ex-012/log/log_test_subject_*.txt \
   result/FV-ex-012/log/prediction.json \
   "result/FV-ex-012/log/seed$SEED/"

echo "Finished at: $(date)"
