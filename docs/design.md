# Orrery design notes

This is the working design document behind the code. The README has the summary and the result tables. This file explains *why* things are built the way they are.

## 1. The question

> How much world (particles, their motion and their interactions) can a single 16 GB consumer GPU hold, if the job is not predicting the next token but predicting cause and effect?

The method is engine-first: build working milestones that each produce something you can watch, and measure them as you go.

## 2. Principles

**Physics in the weights, world in the VRAM.** An LLM's weights hold the rules of language and its context holds the text. Orrery inverts the proportions. The weights are tiny (about 1-2M parameters, because physics has fewer rules than English) and the world state is what fills the card. The scaling axis is world size, not model size.

**Understanding is compression; the laws are the code.** Zoom past the water particles and the useful "core truth" is a handful of generating laws, not smaller particles. Predictive power comes from learning (or injecting) the generator, not from ever finer units. Chaos grows exponentially, and a full-resolution map is as big as the territory.

**Exact spine, learned residual.** Wherever a law is exactly known, compute it analytically and let the network learn only the residual. That covers gravity (uniform or pairwise), Coulomb forces with a trap, and, through the architecture, conservation. Every injected law is one less thing the network can hallucinate. Levitating water (the Gen 1 failure) is the canonical example of an un-injected law.

**Weights are disposable; the engine compounds.** Checkpoints retire every generation; retraining one costs a night of GPU time. What compounds is the chassis (neighbor search, chunked inference, fp16), the corpus pipeline (teacher, trajectories, stats, train), the recipe (noise injection, one-step and unrolled targets) and the measurements.

**The model is the imagination, never the witness.** A learned simulator cannot prove anything about the world. It can only work out consequences of the physics its teachers encoded, plus its own artifacts. Legitimate uses are internal-consistency checks, fast what-if exploration, and ensemble triage (run many variant histories cheaply, then send the interesting ones to an exact solver or a real instrument). A "finding" counts only when a causally independent check confirms it.

**Time is a function of activity.** There is no need for a universal clock. Quiet regions can freeze (no compute means no time passes), while active regions run at full rate. This is the same idea as per-particle timestep rungs in astrophysical N-body codes. Spatial hierarchy (aggregating particles) and time dilation (gating them) are the same compression applied on two axes. `proptime.py` is the first, two-rung version.

## 3. Architecture

```
classical teachers (Taichi MLS-MPM; PyTorch N-body / plasma)
        | trajectories (.npz) + meta.json (radius, dt, domain, law constants)
        v
GNS-style graph network (pure PyTorch, no torch_geometric)
  encoder -> 10 residual GN blocks -> decoder
  node: 5 normalized velocities + clipped wall distances
        (+ temperature history, charge, material / regime embeddings)
  edge: relative displacement / R, distance / R
  target: normalized acceleration MINUS the exact spine
  training: one-step MSE or K-step unrolled loss, random-walk input noise
        |
        v
engine: hash-grid O(N) neighbors, edge-chunked message passing, fp16,
        activity gating, what-if scenes, big-world inference
```

Stack decisions: PyTorch (proven on ROCm and CUDA) plus Taichi with the Vulkan backend for the MPM teachers, which runs natively on AMD. NVIDIA Warp was ruled out because it is CUDA-only, and JAX because its ROCm support was unreliable at the time. Short A/B questions ran on rented cloud GPUs.

## 4. Generations

| Gen | Name | Adds | Status |
|---|---|---|---|
| 1 | Water | the loop exists | done |
| 2 | Honest water | gravity spine, wall features out to 4R, dynamic-frame sampling | done: levitation fixed at the same resolution |
| 3 | Matter | water + jelly + snow, one network, material embedding, 3-step unrolled loss | first version |
| 4 | Heavens | exact N-body gravity spine + learned collisions/accretion; momentum conserved by construction | first version |
| 5 | Flow and heat | temperature channel, buoyancy, convection cells | first version |
| 6 | 3D | third dimension | first version, residual levitation |
| 7 | Property time | activity gating, then dt rungs, then learned dt | two-rung prototype |
| 8 | Hierarchy | super-particles, conservation contracts, refine on demand | not started (the research wall) |
| 9 | Fields | charged particles (plasma); later field grids | first version |
| 10 | One world | several regimes in one network and one scene | mechanism tested; interference not yet measured cleanly (the first measurement used pooled normalization) |

Generations 1-7 are designed to stand on their own, so that Gen 8, the genuinely uncertain one, can stall without sinking the rest.

## 5. Notable failure modes and their fixes

- **Phantom floor (Gen 1).** Wall distances were clipped at R, so the network learned "a dense pile below me means rest" instead of "the floor means rest", and water settled in mid-air. Fixed in Gen 2 with the gravity spine, wall visibility out to 4R and more samples from dynamic frames. The fix stayed at the same resolution.
- **Float-up (Gen 2).** At rest, the learned residual must cancel gravity exactly, and one-step training never feels the error compounding. Fix: the K-step unrolled loss (`--rollout-k`), where the model integrates its own predictions and each step is scored. At k=1 it equals the one-step loss exactly.
- **Sinking cloud (Gen 4).** The free decoder's learned contact forces did not sum to zero, so the predicted cloud's center of mass sank. Fix: the edge-force decoder (`--conserve-momentum`), which conserves momentum for any weights.
- **Edge explosions (Gen 4).** Gravity crushes clumps below the contact diameter, and neighbor counts explode. Fixes: a smaller graph radius (3.5 particle radii), gradient checkpointing, and an optional edge cap that resamples a frame and counts how often it did so, rather than silently dropping dense frames.
- **Thermal teacher (Gen 5).** Three smoke-test failures, each fixed in `gen/mpm_multi.py`:
  - Empty cells read as T=0 and drained all the heat. Fix: no-flux at empty cells.
  - PIC-style temperature replacement acted as a box blur and erased the gradient. Fix: FLIP-style delta transport.
  - A uniformly heated floor reached a static stratified equilibrium. Fix: localized floor heat patches that force circulation.
- **Pooled normalization (Gen 10).** One network trained on two teachers whose velocity and acceleration scales differ 10-24x. Pooled stats crushed one regime's targets and amplified shared output bias, which made particles climb a wall. Fix: per-corpus stats with a regime embedding.
- **ROCm OOMs.** Memory exhaustion on ROCm often surfaces as hipBLAS allocation errors rather than `OutOfMemoryError`. `train.is_oom` catches both, and the trainer drops to a smaller batch instead of crashing.

## 6. Decision log

| Decision | Why |
|---|---|
| GNS lineage over a from-scratch model | proven science; the project builds the engine, not the paper |
| Pure PyTorch, no torch_geometric | avoids ROCm dependency problems; `index_add_` is enough |
| Taichi with the Vulkan backend for the teachers | runs natively on AMD |
| Input noise 6.7e-4 (paper value) | A/B measured: rollout MSE 0.033 vs 0.048 at half the noise |
| `--noise auto` = 0.48 x velocity std | the validated water ratio, made relative to each corpus |
| Law injection over finer resolution | chaos and map-territory limits; Gen 2 confirmed it at fixed resolution |
| K-step unrolled loss (k=3 matter/heat, k=2 3D/N-body) | attacks compounding drift directly; memory sets k |
| Fixed reference temperature in buoyancy | an adaptive mean temperature is global state that a local graph model cannot observe |
| Momentum conservation in the architecture | exact cancellation with no pairing logic; the free decoder demonstrably drifted |
| Gradient checkpointing over capping dense frames | dropping dense frames drops exactly the contact physics; recomputation is cheap |
| Per-corpus normalization when merging teachers | pooled stats caused the wall-climb defect |

## 7. Non-goals

- Operational weather forecasting
- Safety-certified engineering, molecular or quantum fidelity
- Quantitative trust without per-domain calibration
- Proving hypotheses inside the simulator (imagination, not witness)
- Uniform microscopic resolution (refine locally instead)

## 8. Risks

- Cross-scale closure (Gen 8) is genuine research risk.
- Long-horizon drift needs periodic exact correction, not hope.
- Going to 3D divides particle counts by roughly 10 until a hierarchy wins them back.
- Large-world throughput above 100k particles has not been measured since `bigworld.py` moved particle state to fp32. (The original 300k/600k fp16 runs stored positions in fp16, so their numbers are not used.)
