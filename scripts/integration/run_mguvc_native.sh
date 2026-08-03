#!/bin/bash
# scripts/integration/run_mguvc_native.sh
#
# Run mguvc directly, outside Python, to record exactly what it consumes and
# emits. The command below is transcribed from CarpWrapper.run_mguvc and the
# etags content from utilities/uvc.py::write_etags_file, so a run here is the
# same run the library performs — with nothing between you and the output.
#
# Usage:
#   ./run_mguvc_native.sh --lv <tag> --rv <tag> [--data <dir>] [--out <dir>] [--np N]
#
# The LV/RV tags are whatever LabelManager reads from config/labels.yaml:
#   grep -iE 'lv|rv' <data>/config/labels.yaml

set -euo pipefail

DATA_DIR="/data/pycemrg_test_data/ventricular_uvc"
OUT_ROOT=""
LV_TAG=""
RV_TAG=""
NP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lv)   LV_TAG="$2"; shift 2 ;;
        --rv)   RV_TAG="$2"; shift 2 ;;
        --data) DATA_DIR="$2"; shift 2 ;;
        --out)  OUT_ROOT="$2"; shift 2 ;;
        --np)   NP="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$LV_TAG" || -z "$RV_TAG" ]]; then
    echo "ERROR: --lv and --rv are required." >&2
    echo "       Find them with: grep -iE 'lv|rv' $DATA_DIR/config/labels.yaml" >&2
    exit 2
fi

INPUT_MESH_DIR="$DATA_DIR/input_mesh"
[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$DATA_DIR/test"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_ROOT/run_$STAMP"     # everything from this run lives here
WORK_DIR="$RUN_DIR/BiV"            # mesh + vtx together, as mguvc requires
UVC_DIR="$RUN_DIR/uvc"             # must NOT exist before mguvc runs
LOG_FILE="$RUN_DIR/mguvc.log"

# --- Preconditions -----------------------------------------------------

if ! command -v mguvc >/dev/null 2>&1; then
    echo "ERROR: mguvc is not on PATH." >&2
    echo "       Source your CARPentry environment first — the Python path" >&2
    echo "       does this via CarpRunner and PYCEMRG_CARP_CONFIG." >&2
    exit 1
fi

[[ -d "$INPUT_MESH_DIR" ]] || { echo "ERROR: no input_mesh at $INPUT_MESH_DIR" >&2; exit 1; }

# mguvc prompts interactively if its output directory already exists, so it is
# created by mguvc, never by us. This is the trap documented in CLAUDE.md.
[[ -e "$UVC_DIR" ]] && { echo "ERROR: $UVC_DIR already exists; mguvc would prompt." >&2; exit 1; }

mkdir -p "$WORK_DIR"

# --- Stage inputs ------------------------------------------------------
# Mirrors the integration test: copy .pts/.elem plus every .vtx, and nothing
# else. Note there is no BiV.lon in the test data — mguvc evidently does not
# need one. Source data is never written to.

cp "$INPUT_MESH_DIR/BiV.pts"  "$WORK_DIR/"
cp "$INPUT_MESH_DIR/BiV.elem" "$WORK_DIR/"
cp "$INPUT_MESH_DIR"/*.vtx    "$WORK_DIR/"

MESH="$WORK_DIR/BiV"
ETAGS="$WORK_DIR/BiV.etags.sh"

# --- etags -------------------------------------------------------------
# Byte-for-byte what ETagsParameters(mode='base').generate_script_content()
# produces: used tags first, then everything else pinned to UNUSED_TAG=200.

cat > "$ETAGS" <<EOF
#!/bin/bash

## ONLY CHANGE THESE LABELS TO MATCH YOUR MESH LABELS

T_LV=$LV_TAG
T_RV=$RV_TAG

T_UNUSED=200
T_LA=200
T_LABP=200
T_LINFPULMVEINCUT=200
T_LSUPPULMVEINCUT=200
T_RINFPULMVEINCUT=200
T_RSUPPULMVEINCUT=200
T_RA=200
T_RABP=200
T_LVBP=200
T_AORTA=200
T_AORTABP=200
T_MITRALVV=200
T_AORTICVV=200
T_RVBP=200
T_VCINF=200
T_VCSUP=200
T_PULMARTERY=200
T_PULMARTERYBP=200
T_TRICUSPVV=200
T_PULMVV=200
EOF
chmod +x "$ETAGS"

# --- Record the inputs -------------------------------------------------

{
    echo "=== mguvc native run $STAMP ==="
    echo "mguvc:      $(command -v mguvc)"
    echo "data:       $DATA_DIR"
    echo "LV/RV tags: $LV_TAG / $RV_TAG"
    echo
    echo "=== INPUTS staged in $WORK_DIR ==="
    ls -la "$WORK_DIR"
    echo
    echo "=== etags written ==="
    cat "$ETAGS"
    echo
} | tee "$LOG_FILE"

# --- The command -------------------------------------------------------
# Transcribed from CarpWrapper.run_mguvc. Defaults there are
# laplace_solution=True, custom_apex=False, uvc_phi_model="full"; the latter
# two emit no flag at their defaults.

CMD=(
    mguvc
    --model-name    "$MESH"
    --input-model   biv
    --output-model  biv
    --np            "$NP"
    --tags-file     "$ETAGS"
    --output-dir    "$UVC_DIR"
    --laplace-solution
)

{
    echo "=== COMMAND ==="
    printf '%q ' "${CMD[@]}"; echo
    echo
    echo "=== mguvc output ==="
} | tee -a "$LOG_FILE"

set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

# --- Record the outputs ------------------------------------------------

{
    echo
    echo "=== exit status: $STATUS ==="
    echo
    echo "=== EVERYTHING UNDER $RUN_DIR AFTER THE RUN ==="
    if command -v tree >/dev/null 2>&1; then
        tree -a "$RUN_DIR"
    else
        find "$RUN_DIR" | sort
    fi
    echo
    echo "=== files mguvc created in the mesh directory (not the output dir) ==="
    find "$WORK_DIR" -newer "$ETAGS" | sort
} | tee -a "$LOG_FILE"

echo
echo "Log: $LOG_FILE"
exit "$STATUS"
