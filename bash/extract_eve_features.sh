#!/bin/bash
#SBATCH --job-name=eve_features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eve_features_out_%j.log
#
# F4 / Stage B -- per-trial image features for the EVE cohort.
#
# Run it as `bash bash/extract_eve_features.sh`, NEVER `source` it: the -euo pipefail
# below would then apply to the calling shell, and one failed precondition would exit
# the login shell and drop the salloc allocation (TechStack section 1 conventions).
#
#   salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00
#   bash bash/extract_eve_features.sh            # SPLIT=both
#   SPLIT=test bash bash/extract_eve_features.sh # the 1062 scored trials only
#
# `sbatch bash/extract_eve_features.sh --export=ALL,SPLIT=test` works unchanged; the
# #SBATCH block is an inert comment when the script is executed directly.
#
# NO `module load CUDA` is needed, unlike F3: nothing here is compiled. Stage B's
# backbone is torchvision's maskrcnn_resnet50_fpn(...).backbone.body, which shares no
# weights, no code and no init pickles with SE-Net's ImageFeatureEncoder (FR1.1) --
# the Roadmap's "F4 builds the same encoder" note is wrong and this is the correction.
#
# DISK BUDGET (FR5.4). Each tensor is 768 x 2048 x 4 B = 6.29 MB:
#   SPLIT=both -> 1804 files ~ 11.3 GB      SPLIT=test -> 1062 files ~ 6.7 GB
# `test` alone is sufficient for F5 -- test.py never builds a train-split loader
# (TechStack section 3.6) -- and is the economical choice.
#
# THE BUNDLE IS STAGED TO NODE-LOCAL SCRATCH, not read from beegfs. This mirrors
# ~/projects/EyeNet-Pipeline/whole_train.sh, whose config template states the reason
# outright: data paths point at the rsync'd local-scratch copy "to keep training I/O
# off the network filesystem". F4 does 1804 random reads of ~1.3 MB PNGs, which is
# exactly that workload. Note this does NOT contradict TechStack section 1.3's "/tmp
# is node-local" warning: that is about anything which must PERSIST (envs, artefacts,
# the D5 record). A read-only data copy re-staged per job is what scratch is for.
# Everything written -- OUT_DIR, versions.txt, stdout.txt -- still goes to beegfs.
#
# F4 NEEDS A DIFFERENT TAR FROM EyeNet's. `bundle.h5` (0.23 GB) + `stimuli/` (4.0 GB)
# and NOT `face_crops/` (11 GB), which F4 never opens -- get_stimulus() resolves
# samples_df's `stimulus_path` = "stimuli/<exp_key>.png" against the bundle dir, and
# nothing on this path touches a face crop. Build it once on the login node:
#
#   cd <wherever the bundle lives>
#   tar -cf $HOME_DIR/projects/bundle_stimuli.tar bundle/bundle.h5 bundle/stimuli
#
# Set STAGE_BUNDLE=0 to skip staging entirely and point BUNDLE_DIR at an existing
# copy (a beegfs one works, just slower).
#
# Every tunable is overridable from the environment. No absolute path appears in any
# .py (working convention 4).

set -euo pipefail

HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
ISP_ENV="${ISP_ENV:-scanpath}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-/tmp/${USER:-evefeat}}"
STAGE_BUNDLE="${STAGE_BUNDLE:-1}"
BUNDLE_TAR="${BUNDLE_TAR:-$HOME_DIR/projects/bundle_stimuli.tar}"
BUNDLE_DIR="${BUNDLE_DIR:-$LOCAL_SCRATCH/data/bundle}"
BRIDGE_DIR="${BRIDGE_DIR:-$PROJECT_DIR/data/eve_bridge}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/data/eve_features}"
SPLIT="${SPLIT:-both}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
OSIE_EMB="${OSIE_EMB:-$PROJECT_DIR/ISP/OSIE/GazeformerISP/src/data/embeddings.npy}"
MOUNT_IMAGE="${MOUNT_IMAGE:-1}"

case "$SPLIT" in
    both)  NEED_KB=$((12 * 1024 * 1024)); NEED_H="12 GB for 1804 trials" ;;
    test)  NEED_KB=$((7 * 1024 * 1024));  NEED_H="7 GB for 1062 trials" ;;
    train) NEED_KB=$((5 * 1024 * 1024));  NEED_H="5 GB for 742 trials" ;;
    *) echo "FATAL: SPLIT must be both|test|train, got '$SPLIT'" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR" "$PROJECT_DIR/logs"

echo "== F4 Stage B =="
echo "env=$ISP_ENV split=$SPLIT bridge=$BRIDGE_DIR out=$OUT_DIR"
echo "bundle=$BUNDLE_DIR (stage=$STAGE_BUNDLE from $BUNDLE_TAR)"
echo "node=${SLURM_NODELIST:-$(hostname)}"

# ---------------------------------------------------------------------------
# FR9.4 -- the beegfs-side preconditions first, before anything expensive. These are
# the ones that fail on a fresh cluster checkout, because data/ is git-ignored and so
# nothing under it arrives by `git pull` (F3's lesson).
# ---------------------------------------------------------------------------
for path in "$BRIDGE_DIR/fixations.json" "$BRIDGE_DIR/gt_heatmaps.h5" \
            "$BRIDGE_DIR/subject_id_map.json" "$BRIDGE_DIR/bridge_report.json" \
            "$OSIE_EMB"; do
    if [ ! -e "$path" ]; then
        echo "FATAL: $path does not exist." >&2
        echo "data/ is git-ignored, so F2's artefacts must be shipped by hand." >&2
        exit 1
    fi
done

FREE_KB="$(df -Pk "$OUT_DIR" | awk 'NR==2 {print $4}')"
if [ "$FREE_KB" -lt "$NEED_KB" ]; then
    echo "FATAL: $OUT_DIR has ${FREE_KB} KB free; need ~$NEED_H (FR5.4)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# The env image. Non-fatal on purpose, exactly as bash/test_osie.sh has it: under the
# interactive salloc path this script may run several times in ONE allocation, and a
# second mount_image.py exits non-zero, which `set -e` would turn into an abort before
# any precondition ran. `conda activate "$ISP_ENV"` below is the real check.
# ---------------------------------------------------------------------------
if [ "$MOUNT_IMAGE" = "1" ]; then
    echo "-- mounting env image (non-fatal; already-mounted is expected on a re-run)"
    ( cd "$HOME_DIR" && sudo mount_image.py my_env.ext4 --rw ) || \
        echo "   mount_image.py returned $? -- continuing; conda activate is the real check"
else
    echo "-- MOUNT_IMAGE=0, skipping mount"
fi

# ---------------------------------------------------------------------------
# Stage the bundle to node-local scratch. No conda needed, so this happens before
# activation and its failures stay legible.
# ---------------------------------------------------------------------------
if [ "$STAGE_BUNDLE" = "1" ]; then
    [ -f "$BUNDLE_TAR" ] || {
        echo "FATAL: $BUNDLE_TAR does not exist -- build it on the login node with" >&2
        echo "  tar -cf $BUNDLE_TAR bundle/bundle.h5 bundle/stimuli" >&2
        echo "(bundle.h5 + stimuli/ only; face_crops/ is 11 GB F4 never opens)," >&2
        echo "or set STAGE_BUNDLE=0 and point BUNDLE_DIR at an existing copy." >&2
        exit 1
    }
    STAGE_ROOT="$(dirname "$BUNDLE_DIR")"
    mkdir -p "$STAGE_ROOT"
    SCRATCH_FREE_KB="$(df -Pk "$STAGE_ROOT" | awk 'NR==2 {print $4}')"
    TAR_KB="$(du -k "$BUNDLE_TAR" | awk '{print $1}')"
    # tar + its extracted copy coexist during extraction, hence the doubling.
    if [ "$SCRATCH_FREE_KB" -lt $((TAR_KB * 2)) ]; then
        echo "FATAL: $STAGE_ROOT has ${SCRATCH_FREE_KB} KB free; staging needs about" >&2
        echo "$((TAR_KB * 2)) KB (the tar plus its extracted copy)." >&2
        exit 1
    fi
    if [ -f "$BUNDLE_DIR/bundle.h5" ]; then
        echo "-- bundle already staged at $BUNDLE_DIR; skipping rsync/extract"
    else
        echo "-- staging $BUNDLE_TAR -> $STAGE_ROOT (keeps 1804 PNG reads off beegfs)"
        rsync -ah --progress "$BUNDLE_TAR" "$STAGE_ROOT/"
        tar -xf "$STAGE_ROOT/$(basename "$BUNDLE_TAR")" -C "$STAGE_ROOT/" \
            --checkpoint=2000 --checkpoint-action=echo="   extracted %u files"
        rm -f "$STAGE_ROOT/$(basename "$BUNDLE_TAR")"
    fi
fi

for path in "$BUNDLE_DIR" "$BUNDLE_DIR/bundle.h5" "$BUNDLE_DIR/stimuli"; do
    if [ ! -e "$path" ]; then
        echo "FATAL: $path does not exist after staging." >&2
        echo "The tar must expand to a 'bundle/' directory holding bundle.h5 and" >&2
        echo "stimuli/ -- get_stimulus() resolves 'stimuli/<exp_key>.png' against it." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Activation. The conda.sh path is EXPLICIT, not `$(conda info --base)`: before any
# env is active conda need not be on PATH at all, and the substitution would fail
# first. bash/test_osie.sh and EyeNet-Pipeline/whole_train.sh both spell it out.
#
# `set -u` is lifted ONLY across conda activate and restored at once: an env whose
# etc/conda/activate.d hook reads a variable before assigning it (senet's
# libblas_mkl_activate.sh does) would otherwise abort the whole script before a line
# of our code runs -- TechStack section 1.3's fourth cluster fact. `scanpath` carries
# no such hook, but ISP_ENV is a tunable, so the guard stays. -e and -o pipefail are
# never lifted, so a genuine activation failure still stops the run.
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
# shellcheck disable=SC1091
set +u
source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$ISP_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR/ISP/OSIE/GazeformerISP/src:${PYTHONPATH:-}"

{
    # -----------------------------------------------------------------------
    # FR1.2 / FR1.4 / FR1.5 -- the version probe. One resolution, printed here and
    # embedded in feature_report.json, so the record cannot disagree with the check
    # (TechStack section 1.1's lesson from F1's wrong skimage/opencv row).
    # evedataset is imported WHOLE (FR1.3): it pulls torchvision.transforms.v2, so a
    # torchvision too old for v2 must fail at preflight rather than mid-run.
    # -----------------------------------------------------------------------
    python - "$OUT_DIR/versions.txt" <<'PY'
import json, sys
sys.path.insert(0, "tools")
from eve_prep.extract_features import require_environment, versions

resolved = versions()
with open(sys.argv[1], "w") as fh:
    json.dump(resolved, fh, indent=2)
print(json.dumps(resolved, indent=2))
require_environment(resolved, require_cuda=True)
PY

    # -----------------------------------------------------------------------
    # FR9.6 -- extraction is guarded by the preflight's exit code, so a complete
    # cache costs nothing on a re-run. FORCE_FEATURES=1 overrides.
    # -----------------------------------------------------------------------
    extract() {
        CUDA_VISIBLE_DEVICES=0 python tools/eve_prep/extract_features.py \
            --bundle-dir      "$BUNDLE_DIR" \
            --bridge-dir      "$BRIDGE_DIR" \
            --out-dir         "$OUT_DIR" \
            --split           "$SPLIT" \
            --osie-embeddings "$OSIE_EMB" "$@"
    }

    if [ "$FORCE_FEATURES" = "1" ]; then
        echo "FORCE_FEATURES=1 -- re-extracting every trial."
        extract --overwrite
    elif ! python tools/eve_prep/check_features.py \
            --fix "$BRIDGE_DIR/fixations.json" \
            --heatmaps "$BRIDGE_DIR/gt_heatmaps.h5" \
            --feat-dir "$OUT_DIR/image_features" \
            --split "$SPLIT" ; then
        extract
    else
        # The cache is complete, so no tensor is recomputed -- but the extractor is
        # still re-entered, because FR8.3 says a fully-skipped run must leave a
        # record (D5). Every trial takes the skip branch and the report is rewritten
        # from the cache; this costs seconds, not GPU hours.
        echo "Feature cache already complete for split=$SPLIT; no tensor will be"
        echo "recomputed. Rewriting feature_report.json (FR8.3):"
        extract
    fi

    # -----------------------------------------------------------------------
    # FR9.6 -- the post-check, whose failure fails the job. A deleted .pth must
    # never let this script exit 0.
    # -----------------------------------------------------------------------
    python tools/eve_prep/check_features.py \
        --fix "$BRIDGE_DIR/fixations.json" \
        --heatmaps "$BRIDGE_DIR/gt_heatmaps.h5" \
        --feat-dir "$OUT_DIR/image_features" \
        --split "$SPLIT"

    echo "== F4 done: $OUT_DIR (split=$SPLIT) =="
# FR9.7 -- the interactive path has no SLURM log, so stdout is an artefact.
# `set -o pipefail` is what makes this safe: without it the tee's exit status would
# mask a failed post-check and the job would report success on an incomplete cache.
} 2>&1 | tee "$OUT_DIR/stdout.txt"
