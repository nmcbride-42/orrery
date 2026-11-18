#!/usr/bin/env bash
# The corpus + training flags behind each generation's reported result
# (README "Reported historical results"). GPU-scale: hours to a night each.
#
#   bash scripts/recipes.sh gen2            # generate corpus (if missing), train, evaluate
#   STOP_AT=0 bash scripts/recipes.sh gen5  # run the whole schedule instead
#   PYTHON=.venv/bin/python DEVICE=cuda bash scripts/recipes.sh gen4
#
# Step horizon. train.py decays the learning rate over --steps, so the LR at
# any given step depends on the horizon, not only on the step. Gen 1 and Gen 2
# (and the noise A/B) ran their full 200k schedule. The cloud runs (Gen 3, 4,
# 5, 6, 9 and the merge run) were all launched with --steps 600000 and stopped
# by a wall-clock budget well before that. Each recipe below sets that horizon
# (STEPS) and the step the reported checkpoint came from (STOP_AT), so the LR
# schedule matches. Checkpoints were written every 1000 steps, so a stop step
# is accurate to about +-1000 steps.
#
# What these recipes do NOT reproduce exactly is noted per recipe (gen4free,
# merge). Hardware, nondeterministic GPU kernels and corpus RNG also differ
# from the original runs, so expect the same behaviour, not the same digits.
#
# Training is resumable: re-running the same recipe continues from
# runs/<gen>/latest.pt. Evaluation renders runs/<gen>/rollout_<name>.gif.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RUN=${1:?usage: recipes.sh gen1|gen1lownoise|gen2|gen3|gen4|gen4free|gen5|gen6|gen9|merge}
DEVICE=${DEVICE:-cuda}
OUT=runs/$RUN

gen_water2d()  { [ -f data/water2d/meta.json ]  || "$PY" -m orrery.gen.mpm2d --out data/water2d --traj 40 --particles 3000 --frames 400; }
gen_multi2d()  { [ -f data/multi2d/meta.json ]  || "$PY" -m orrery.gen.mpm_multi --out data/multi2d --traj 60 --pure-water-traj 8; }
gen_thermal()  { [ -f data/thermal2d/meta.json ] || "$PY" -m orrery.gen.mpm_multi --thermal --out data/thermal2d --traj 30; }
gen_water3d()  { [ -f data/water3d/meta.json ]  || "$PY" -m orrery.gen.mpm3d --out data/water3d --traj 40; }
gen_nbody()    { [ -f data/nbody2d/meta.json ]  || "$PY" -m orrery.gen.nbody --out data/nbody2d --traj 30 --particles 2000 --frames 400 --device "$DEVICE"; }
gen_charged()  { [ -f data/charged2d/meta.json ] || "$PY" -m orrery.gen.charged --out data/charged2d --traj 30 --particles 2000 --frames 400 --device "$DEVICE"; }

# EVALS: space-separated data_dir:traj_index:regime:name (regime -1 = single-corpus ckpt)
case $RUN in
  gen1)   # everything learned: no spine, walls visible only within R
    gen_water2d; DATA=data/water2d; H=200000; STOP=0
    ARGS=""
    EVALS="data/water2d:39:-1:water" ;;
  gen1lownoise)  # input-noise A/B arm: Gen 1 with half the paper's noise
    gen_water2d; DATA=data/water2d; H=200000; STOP=0
    ARGS="--noise 3.35e-4"
    EVALS="data/water2d:39:-1:water" ;;
  gen2)   # exact gravity spine, wall features out to 4R, half the samples from dynamic frames
    gen_water2d; DATA=data/water2d; H=200000; STOP=0
    ARGS="--gravity-residual --wall-clip 4.0 --dynamic-frac 0.5"
    EVALS="data/water2d:39:-1:water" ;;
  gen3)   # water/jelly/snow in one net; holdout 10 = 2 mixed + the 8 pure-water scenes
    gen_multi2d; DATA=data/multi2d; H=600000; STOP=131000
    ARGS="--rollout-k 3 --gravity-residual --wall-clip 4.0 --dynamic-frac 0.3 --holdout 10"
    EVALS="data/multi2d:59:-1:mixed data/multi2d:67:-1:purewater" ;;
  gen4)   # N-body: gravity spine + momentum-conserving edge-force decoder
    gen_nbody; DATA=data/nbody2d; H=600000; STOP=65000
    ARGS="--rollout-k 2 --noise auto --conserve-momentum --grad-checkpoint --holdout 2 --max-edges 500000"
    EVALS="data/nbody2d:29:-1:heavens" ;;
  gen4free)  # N-body with the free (non-conserving) decoder.
    # Best reconstruction only: the original free-decoder run's flags were
    # patched on the cloud instance and not fully recorded (its unroll depth
    # and the teacher's graph radius may differ from this). This is the
    # controlled comparison: gen4 minus --conserve-momentum.
    gen_nbody; DATA=data/nbody2d; H=600000; STOP=95000
    ARGS="--rollout-k 2 --noise auto --grad-checkpoint --holdout 2 --max-edges 500000"
    EVALS="data/nbody2d:29:-1:heavens" ;;
  gen5)   # thermal convection: temperature channel + buoyancy learned on top of the gravity spine
    gen_thermal; DATA=data/thermal2d; H=600000; STOP=74000
    ARGS="--rollout-k 3 --gravity-residual --wall-clip 4.0 --holdout 2"
    EVALS="data/thermal2d:29:-1:convection" ;;
  gen6)   # 3D water (the reported run was resumed once, with the same horizon)
    gen_water3d; DATA=data/water3d; H=600000; STOP=251000
    ARGS="--rollout-k 2 --gravity-residual --wall-clip 4.0 --dynamic-frac 0.3 --holdout 2"
    EVALS="data/water3d:39:-1:water3d" ;;
  gen9)   # two-species plasma: Coulomb + trap spine, conserving decoder, charge as a node feature
    gen_charged; DATA=data/charged2d; H=600000; STOP=241000
    ARGS="--rollout-k 2 --noise auto --conserve-momentum --grad-checkpoint --holdout 2 --max-edges 500000"
    EVALS="data/charged2d:29:-1:plasma" ;;
  merge)  # one net, two teachers, regime embedding, per-corpus normalization.
    # NOT the run behind the README's merge row: that run used an earlier
    # version of train.py with pooled normalization (one stats dict for both
    # corpora), which this code no longer has. This recipe runs the current,
    # per-corpus code with the same flags and horizon; it has no reported result.
    gen_multi2d; gen_thermal; DATA=data/multi2d,data/thermal2d; H=600000; STOP=96000
    ARGS="--rollout-k 3 --gravity-residual --wall-clip 4.0 --dynamic-frac 0.3 --holdout 10,2"
    EVALS="data/multi2d:67:0:purewater data/multi2d:59:0:mixed data/thermal2d:29:1:convection" ;;
  *) echo "unknown recipe: $RUN" >&2; exit 1 ;;
esac
STEPS=${STEPS:-$H}
STOP_AT=${STOP_AT:-$STOP}

mkdir -p "$OUT"
# shellcheck disable=SC2086
"$PY" -m orrery.train --data "$DATA" --out "$OUT" --batch 2 --device "$DEVICE" \
    --steps "$STEPS" --stop-at "$STOP_AT" $ARGS --resume 2>&1 | tee -a "$OUT/train.log"

for spec in $EVALS; do
    IFS=: read -r dir idx reg name <<< "$spec"
    "$PY" -m orrery.rollout --ckpt "$OUT/latest.pt" --data "$dir" --traj "$idx" \
        --regime "$reg" --steps 300 --metrics --device "$DEVICE" \
        --gif "$OUT/rollout_$name.gif" 2>&1 | tee -a "$OUT/eval.log"
done
