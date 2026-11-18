#!/usr/bin/env bash
# CPU smoke run of the whole loop: teacher -> corpus -> train -> rollout GIF
# -> what-if GIF. Measured at about 25 minutes on a shared 2-vCPU VM.
# The model is far too small and too briefly trained to be any good:
# the point is that every stage runs. Lower STEPS to go faster.
#
#   bash scripts/smoke.sh            # outputs land in out/smoke/
#   PYTHON=.venv/bin/python bash scripts/smoke.sh
#
# Uses the Taichi water teacher when taichi is installed (pip install -e
# ".[mpm]"), otherwise falls back to the pure-torch N-body teacher.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${OUT:-out/smoke}
STEPS=${STEPS:-300}
mkdir -p "$OUT"

if "$PY" -c "import taichi" >/dev/null 2>&1; then
    KIND=water
    "$PY" -m orrery.gen.mpm2d --out "$OUT/data" --traj 6 --particles 1000 \
        --frames 120 --arch cpu
    TRAIN_ARGS=(--gravity-residual --wall-clip 4.0 --dynamic-frac 0.5)
else
    KIND=nbody
    echo "[smoke] taichi not installed; using the N-body teacher"
    "$PY" -m orrery.gen.nbody --out "$OUT/data" --traj 6 --particles 400 \
        --frames 120 --substeps-per-frame 10 --device cpu
    TRAIN_ARGS=(--noise auto --conserve-momentum)
fi

"$PY" -m orrery.train --data "$OUT/data" --out "$OUT/run" --steps "$STEPS" \
    --batch 2 --hidden 64 --layers 4 --holdout 1 --log-every 50 \
    --ckpt-every 100 --device cpu "${TRAIN_ARGS[@]}"

"$PY" -m orrery.rollout --ckpt "$OUT/run/latest.pt" --data "$OUT/data" \
    --steps 100 --metrics --gif "$OUT/rollout.gif" --device cpu

if [ "$KIND" = water ]; then
    "$PY" -m orrery.whatif --ckpt "$OUT/run/latest.pt" --data "$OUT/data" \
        --scene dam_break --steps 60 --gif "$OUT/whatif.gif" --device cpu
fi
echo "[smoke] done: see $OUT/"
