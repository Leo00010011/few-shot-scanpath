#!/bin/bash
#SBATCH --job-name=osie_eval
#SBATCH --output=logs/osie_out_%j.log
#SBATCH --error=logs/osie_err_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal
#
# OSIE few-shot query-set evaluation -- Roadmap F1 baseline.
# Spec: spec/2026-09-08-osie-eval-baseline/
#
# PRIMARY PATH -- interactive, under salloc. Easier to debug, and the way the F1
# sweep is actually being run:
#
#   salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=08:00:00
#   bash bash/test_osie.sh                     # SEED=0
#   SEED=1 bash bash/test_osie.sh
#   SEED=2 bash bash/test_osie.sh
#
#   *** `bash script`, never `source script`. ***
#   `set -euo pipefail` below is scoped to the script when it is executed, but with
#   `source` it applies to YOUR shell -- one failed preflight and the allocation's
#   login shell exits, taking the allocation with it.
#
# The #SBATCH block above is retained so `sbatch bash/test_osie.sh` (with
# `--export=ALL,SEED=1` for the other seeds) still works unchanged. Nothing in the
# body depends on which path is used: the directives are inert comments when the
# script is executed directly, SLURM_NODELIST has a fallback, and SEED is read from
# the environment either way.
#
# One difference that matters when running manually: there is no
# logs/osie_out_<jobid>.log. Everything lands in your terminal. That is precisely
# why test.py's stdout is tee'd into log/seed$SEED/stdout.txt further down -- the
# headline SM/MM/SED line is a bare print(), and scrollback is not an artefact.
#
# Retargeted 2026-09-09 from bash/test_cocofv.sh when F1 reverted to OSIE.
# Every tunable below is overridable from the environment, so a variant run needs
# no edit to the script body (FR8.4).

set -euo pipefail

echo "Starting OSIE eval at: $(date)"
echo "Running on node: ${SLURM_NODELIST:-<not under slurm>}"

# ---- tunables (FR8.4) ------------------------------------------------------
HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
# A FLAT directory of the 700 OSIE 800x600 .jpg stimuli, 1001.jpg .. 1700.jpg
# (NUS-VIP predicting-human-gaze-beyond-pixels). Stage B only -- see the note below.
# Staged 2026-09-09 at $PROJECT_DIR/data/stimuli. That is inside the checkout, which
# is safe only because data/ is git-ignored (working convention 5, TechStack 6.5) --
# never `git add -f` anything under it.
OSIE_IMAGE_ROOT="${OSIE_IMAGE_ROOT:-$PROJECT_DIR/data/stimuli}"
OSIE_WORK="${OSIE_WORK:-$PROJECT_DIR/work/osie}"   # BeeGFS -- features stream from here
OSIE_DATA="${OSIE_DATA:-$OSIE_WORK/data}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$PROJECT_DIR/weights/OSIE-20260904T121550Z-1-001/OSIE}"
ISP_ENV="scanpath"
FEWSHOT_SUBJECTS="${FEWSHOT_SUBJECTS:-10 11 12 13 14}"
SUBJECT_NUM="${SUBJECT_NUM:-5}"
SEED="${SEED:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-/tmp/${USER:-osie}}"
# ---- end tunables ----------------------------------------------------------

BRANCH_DIR="$PROJECT_DIR/ISP/OSIE/GazeformerISP"
PREP="$PROJECT_DIR/tools/osie_prep"
EVAL_DIR="src/assets/OSIE-ex-10to15"

# The OSIE branch SHIPS its own canonical labels and task embedding, so unlike the
# COCO-FreeView script there is no normalisation step and no hub access at all.
#
#   *** Do NOT point --fix_dir at data/osie_fixations_update_duration.json. ***
#
# That file has identical X/Y and identical keys, but its T is the decile BIN INDEX
# (0-9) of the duration rather than the duration in ms. It would not crash anything;
# it would silently invalidate ScanMatch-with-duration and MultiMatch's duration
# dimension, and therefore both headline SM and MM. check_fixations invariant (g)
# catches it, and this default avoids it. (Verified 2026-09-09.)
FIX_JSON="${FIX_JSON:-$BRANCH_DIR/src/data/fixations.json}"
EMB_NPY="${EMB_NPY:-$BRANCH_DIR/src/data/embeddings.npy}"

cd "$HOME_DIR"
# Non-fatal on purpose. Under the interactive salloc path the script is run several
# times inside ONE allocation (three seeds, plus debug re-runs), and the image is
# already mounted from the first invocation -- a second mount_image.py exits non-zero
# and `set -e` would abort the run before a single precondition had been checked.
# Skipping the guard costs nothing: if the mount genuinely did not happen, the
# `conda activate "$ISP_ENV"` below fails loudly and immediately, which is the real
# check. Set MOUNT_IMAGE=0 to skip the attempt entirely.
MOUNT_IMAGE="${MOUNT_IMAGE:-1}"
if [ "$MOUNT_IMAGE" = "1" ]; then
  echo "Mounting image (non-fatal; already-mounted is expected on a re-run)"
  sudo mount_image.py my_env.ext4 --rw || \
    echo "  mount_image.py returned $? -- continuing; conda activate is the real check"
else
  echo "MOUNT_IMAGE=0, skipping mount"
fi

# ---- FR2.4: cheap preconditions before anything expensive ------------------
if [ ! -d "$OSIE_IMAGE_ROOT" ]; then
  echo "FATAL: OSIE_IMAGE_ROOT does not exist: $OSIE_IMAGE_ROOT" >&2; exit 1
fi
if [ -z "$(find "$OSIE_IMAGE_ROOT" -maxdepth 1 -name '*.jpg' -print -quit)" ]; then
  echo "FATAL: OSIE_IMAGE_ROOT has no *.jpg directly under it: $OSIE_IMAGE_ROOT" >&2
  echo "       OSIE's stimulus directory is FLAT -- no <category>/ subdirectories." >&2
  exit 1
fi
if [ ! -f "$FIX_JSON" ]; then
  echo "FATAL: FIX_JSON does not exist: $FIX_JSON" >&2; exit 1
fi
if [ ! -f "$EMB_NPY" ]; then
  echo "FATAL: EMB_NPY does not exist: $EMB_NPY" >&2; exit 1
fi

# ---- FR2: checkpoint + subject embedding, by observation (FR6) -------------
# TechStack 3.4 warns about the case inconsistency in the READMEs
# (OSIE-ex-10to15 vs osie-ex-10to15, user_ vs subject_). On a case-sensitive
# filesystem that is a real failure mode, so enumerate before wiring.
echo "--- on-disk names under $WEIGHTS_DIR ---"
ls -la "$WEIGHTS_DIR"
CKPT="$WEIGHTS_DIR/checkpoint_best.pth"
USER_EMB="${USER_EMB:-$WEIGHTS_DIR/fewshot_user_embedding_10.pt}"
for f in "$CKPT" "$USER_EMB"; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
mkdir -p "$BRANCH_DIR/$EVAL_DIR/checkpoints"
ln -sf "$CKPT" "$BRANCH_DIR/$EVAL_DIR/checkpoints/checkpoint_best.pth"

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$ISP_ENV"

# ---- FR1.2: resolved versions into the SLURM log AND into a kept artefact --
# D5 requires every run to be self-describing, and TechStack section 1.1 is blunt
# about why this one matters: F1's green run used an env several major versions
# newer than ISP/environment.yml, and "consistent with the published numbers" is a
# coarser test than bitwise equality. A future discrepancy can only be attributed
# if each run's resolved stack was written down. The SLURM log is named by job id,
# not by seed, so the version dump is also teed into a file the per-seed copy step
# below preserves.
mkdir -p "$OSIE_WORK"
python - <<'PY' | tee "$OSIE_WORK/versions.txt"
import importlib, sys
print("python", sys.version.replace("\n", " "))
for m in ("torch", "numpy", "scipy", "skimage", "multimatch_gaze", "cv2"):
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "MISSING", e)
PY

# ---- FR1.3: multimatch-gaze is metric-critical, pinned exactly --------------
# `python -m pip`, not `py -m pip`: the `py` launcher is Windows-only and does not
# exist on the cluster (TechStack section 1 command conventions).
#
# Install only if the pin is not already satisfied, and with --no-deps. $ISP_ENV may
# be a pre-existing environment shared with other work (it is a tunable, and reusing
# one is supported): an unconditional `pip install` lets pip resolve multimatch-gaze's
# numpy/scipy/pandas requirements and silently move versions the rest of that env
# depends on. The verification below is what actually enforces the pin -- so if the
# --no-deps install leaves an unimportable package, this stops before the GPU is used.
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
        "FATAL: multimatch-gaze {} != 0.1.3 (metric-critical pin, TechStack section 1);"
        " the numbers would not be comparable to the published table\n".format(v))
    sys.exit(1)
print("multimatch-gaze 0.1.3 OK")
PY

# ---- FR6.6: the checkpoint must not carry a baked-in subject embedding ------
# gazeformer.py loads self.subject_embed from --user_emb_path; a surviving
# subject_embed.weight in the state dict would mean the checkpoint's own base-set
# table is in play instead of the query subjects'.
python - "$CKPT" "$USER_EMB" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
keys = [k for k in getattr(sd, "keys", lambda: [])() if "subject_embed" in k]
if keys:
    sys.stderr.write("FATAL: checkpoint carries {} (FR6.6)\n".format(keys))
    sys.exit(1)
print("checkpoint has no subject_embed key OK")

emb = torch.load(sys.argv[2], map_location="cpu").float()
print("user embedding", tuple(emb.shape))
assert emb.shape[1] == 384, "subject_feature_dim != 384"
# select_fewshot_subject() remaps --fewshot_subject 10..14 to a dense 0..4 and
# gazeformer indexes subject_embed[subjects], so ONLY rows 0-4 are read. Verified
# on the dev machine 2026-09-09: rows 5-9 of fewshot_user_embedding_10.pt are
# exactly zero. Re-asserted here because a silently different tensor on the
# cluster (e.g. train_user_embedding.pt copied into place) would score the query
# subjects against base-set embeddings and still produce a plausible number.
nz = [i for i in range(emb.shape[0]) if float(emb[i].abs().sum()) != 0.0]
print("non-zero embedding rows:", nz)
PY

# ---- FR3.5 preflight; stdout is the counter JSON, stderr is the commentary --
cd "$PROJECT_DIR"
mkdir -p "$OSIE_WORK"
python "$PREP/check_fixations.py" \
    --fix "$FIX_JSON" \
    --images "$OSIE_IMAGE_ROOT" \
    --fewshot-subject $FEWSHOT_SUBJECTS \
    --split test \
    | tee "$OSIE_WORK/preflight_fixations.json"

# ---- Stage B, guarded (FR8.6) ----------------------------------------------
# Two upstream quirks are worked around at the CALL SITE, never in the file
# (working convention 2, additive over invasive):
#   1. image_data() reads a hardcoded "<dataset_path>/train/" subdirectory, so a
#      symlink farm gives it that view without moving 700 images.
#   2. feature_extractor.py's __main__ crashes on `text_data(dataset_path=args.p)`
#      -- args.p is never defined. image_data() is therefore imported and called
#      directly. text_data() is not needed: the branch ships embeddings.npy with
#      the "free-viewing" key already (verified (768,) float32).
mkdir -p "$OSIE_DATA/image_features" "$LOCAL_SCRATCH/osie/stimuli"
STAGE_B_ROOT="$LOCAL_SCRATCH/osie/stimuli"
ln -sfn "$OSIE_IMAGE_ROOT" "$STAGE_B_ROOT/train"

cd "$BRANCH_DIR"
if [ "$FORCE_FEATURES" = "1" ] || \
   ! python "$PREP/check_features.py" --fix "$FIX_JSON" \
                                      --feat-dir "$OSIE_DATA/image_features" ; then
  echo "Extracting image features into $OSIE_DATA/image_features"
  PYTHONPATH="$BRANCH_DIR/src" python - "$STAGE_B_ROOT" "$OSIE_DATA" <<'PY'
import sys, types, torch

# feature_extractor.py does `from sentence_transformers import SentenceTransformer`
# at MODULE scope, but the name is used only at line 63, inside text_data() -- which
# this run never calls, because the branch ships embeddings.npy with the
# "free-viewing" key already. Rather than pull sentence-transformers (and with it
# transformers + tokenizers, and pip's opinion about which torch to have) into
# $ISP_ENV just to satisfy an import, register a stub first. Call-site only; the
# upstream file is untouched (working convention 2).
if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401  -- real one present, use it
    except ImportError:
        _stub = types.ModuleType("sentence_transformers")
        class _Unavailable:                     # pragma: no cover - never constructed
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "sentence_transformers is stubbed: text_data() is not part of "
                    "the OSIE run (embeddings.npy ships with the branch). Install "
                    "the real package if you need to regenerate task embeddings.")
        _stub.SentenceTransformer = _Unavailable
        sys.modules["sentence_transformers"] = _stub
        print("sentence_transformers stubbed (text_data() unused)")

from preprocess.feature_extractor import image_data
image_data(dataset_path=sys.argv[1], output_path=sys.argv[2],
           device=torch.device("cuda:0"), overwrite=False)
PY
  python "$PREP/check_features.py" --fix "$FIX_JSON" \
                                   --feat-dir "$OSIE_DATA/image_features"
else
  echo "Feature cache complete; skipping extraction (FORCE_FEATURES=1 to override)"
fi

# ---- Stage D + E (FR7.1, FR7.2) --------------------------------------------
# cd'd into $BRANCH_DIR above: every relative default in test.py is anchored to
# ISP/OSIE/GazeformerISP/ (TechStack section 1). The preflight tools are called by
# absolute path so they work from either directory.
#
# $FEWSHOT_SUBJECTS is intentionally UNQUOTED: argparse nargs='+' needs five
# separate words. Quoting it yields fewshot_subject: ['10 11 12 13 14'] and
# argparse's type=int then fails. Do not "fix" this.                    (FR8.4)
#
# --width 512 --height 384 are passed explicitly even though they are the defaults:
# they ARE the metric screen (TechStack section 4), and a silent change would
# rescale every coordinate and re-shape the ScanMatch bins. origin_size is NOT
# passed -- test.py does not forward it and OSIE_evaluation's (600, 800) default is
# already correct for OSIE's native stimuli (D3).
#
# The `if i_batch > 100: break` cap in test.py is left UNMODIFIED and is inert
# here: __len__ returns the number of IMAGES (70 on the test split), not records,
# so the loader yields 70 batches at --batch 1 and 18 at the default --batch 4.
# The shipped prediction.json's 350 records = 70 images x 5 subjects confirms the
# published number covers the full split. (Roadmap F1, closed 2026-09-09.)
#
# stdout is TEE'd, and that is not cosmetic. The headline `SM / MM / SED` line is a
# bare print() on the last line of main(), so it lands on stdout and NEVER in
# log_test_subject_*.txt (TechStack section 3.5a note 1). The SLURM log that would
# otherwise be its only home is named by JOB ID, not by seed, so after a three-seed
# sweep the three headline numbers sit in three logs/osie_out_<jobid>.log files with
# nothing but submission order to tell them apart. Teeing into the seed directory
# makes each seed's headline an artefact the sweep owns.
#
# The logger's StreamHandler writes to stderr, so the metric lines and tqdm stay out
# of this file; it captures the version dump, the preflight commentary and the
# headline. `set -o pipefail` is on, so test.py's exit status still fails the job.
RESULT_LOG="result/$(basename "$EVAL_DIR")/log"
SEED_DIR="$RESULT_LOG/seed$SEED"
mkdir -p "$SEED_DIR"

CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject $FEWSHOT_SUBJECTS \
  --subject_num "$SUBJECT_NUM" \
  --seed "$SEED" \
  --eval_repeat_num 1 \
  --width 512 --height 384 \
  --fix_dir  "$FIX_JSON" \
  --feat_dir "$OSIE_DATA/image_features" \
  --emb_dir  "$EMB_NPY" \
  --img_dir  "$OSIE_IMAGE_ROOT" \
  --evaluation_dir "$EVAL_DIR" \
  --user_emb_path "$USER_EMB" \
  | tee "$SEED_DIR/stdout.txt"

# ---- FR15.2: preserve per-seed artefacts before the next seed overwrites ----
# All three seeds write the same log_test_subject_10_0.txt (the filename tracks
# --num_fewshot/--random_support, neither of which affects a query-set score:
# select_fewshot_subject() returns early when split != 'train'). test.py truncates
# that file at the top of main(), so each seed's copy holds exactly one run's block
# despite the FileHandler's mode='a'.
#
# Four artefacts per seed, which is what tools/osie_prep/aggregate_seeds.py needs to
# pool the runs without guessing: the metric block + resolved arg namespace, the
# generated scanpaths, the headline line, and the resolved environment (D5).
cp "$RESULT_LOG"/log_test_subject_*.txt \
   "$RESULT_LOG/prediction.json" \
   "$SEED_DIR/"
cp "$OSIE_WORK/versions.txt" "$SEED_DIR/versions.txt"
cp "$OSIE_WORK/preflight_fixations.json" "$SEED_DIR/preflight_fixations.json"

echo "Preserved seed $SEED artefacts under $BRANCH_DIR/$SEED_DIR:"
ls -la "$SEED_DIR"

# ---- FR13.3: aggregate once every seed has landed --------------------------
# Not run here -- a single invocation only ever holds its own seed. After the last of
#   bash bash/test_osie.sh                        # SEED=0
#   SEED=1 bash bash/test_osie.sh
#   SEED=2 bash bash/test_osie.sh
# finishes, from $BRANCH_DIR (stdlib only, no GPU -- so it also runs on a login node
# once the allocation is released, or on the Windows checkout with `py`):
#   python "$PROJECT_DIR/tools/osie_prep/aggregate_seeds.py" \
#       --log-dir "$RESULT_LOG" --out "$RESULT_LOG/metrics_sweep.json" \
#       --markdown "$RESULT_LOG/metrics_sweep.md"
# It cross-checks that the pooled runs really are replicates and emits the D6 block
# with an ACROSS-SEED spread. That spread is not cur_metrics_std -- see the module
# docstring; the per-cell std stays deferred to F6.
echo "Seeds present so far: $(ls -d "$RESULT_LOG"/seed* 2>/dev/null | tr '\n' ' ')"

echo "Finished at: $(date)"
