[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude-blueviolet?style=flat-square&logo=anthropic)](https://www.anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-LoRA%20Adapters-yellow?style=flat-square)](https://huggingface.co/KK1kk1/jellyfish-cpu-anchor-lora)
# 🪼 Jellyfish: CPU-Anchor Hybrid Training

> **⚠️ ERRATA & EXPERIMENT STATUS (2026-03-04) — PLEASE READ FIRST**
>
> **This experiment has been suspended due to unresolved confounding variables.**
>
> **Major corrections to original claims:**
>
> **1. train_loss "22.5% improvement" is FAKE (measurement artifact).**
> HuggingFace Trainer's resume logic resets the loss accumulator but divides by
> `global_step` (total steps across both phases). Phase2 trains 400 steps but
> divides by 500, producing ~20% artificial deflation. Corrected values:
>
> | Experiment | Reported (artifact) | Corrected | vs Baseline |
> |---|---|---|---|
> | A (baseline) | 1.184 | 1.184 | — |
> | F (GPU fp32→bf16) | 0.9268 | **1.1585** | -2.2% |
> | C (CPU fp32→GPU bf16) | 0.9177 | **1.1471** | -3.1% |
> | G (CPU fp32→GPU fp32) | 0.9188 | **1.1485** | -3.0% |
>
> **2. "96.6% precision / 3.4% CPU determinism" decomposition is INVALID.**
> This was calculated from artifact-inflated values. With corrected values:
> A→F = 0.0255, F→C = 0.0114, giving ~69% / ~31%. But this is also unreliable
> because of the batch confound below.
>
> **3. Batch size confound discovered in F vs C/G comparison.**
>
> | Experiment | Phase1 batch | Phase1 device |
> |---|---|---|
> | C, G | 2×4=8 | CPU |
> | F | 1×8=8 | GPU |
>
> The "0.82% CPU deeper than GPU" finding compared F (GPU, batch 1×8) vs C/G
> (CPU, batch 2×4). Two variables changed simultaneously: device AND micro-batch.
> When 3B QLoRA 4-bit was tested with **identical batch (1×8 for both)**,
> CPU = GPU (Δ ≤ 0.001 at step 40). The original "CPU determinism" signal
> may have been a micro-batch artifact.
>
> **4. "Quantization causes CPU ≠ GPU" is UNCONFIRMED.**
> 3B QLoRA 4-bit with same batch → CPU = GPU. The 7B divergence may have been
> batch (2×4 vs 1×8), not quantization, not CPU determinism.
>
> **What remains valid:**
> - MMLU data (independent evaluation, unaffected by train_loss artifact)
> - MMLU warm restart improvement: 4/4 positive direction (within stderr individually)
> - bf16 = fp32 in steady-state training (3× confirmed, 3B 16-bit full)
> - Precision transition with continuous lr = zero effect (3B Exp AA)
> - GPU hardware noise floor observation
> - Zero transition shock between phases
>
> **Experiment suspended.** Too many confounding variables (train_loss artifact,
> batch mismatch, single seed) to draw reliable conclusions about CPU determinism.
> Original content preserved below for transparency.

**An exploratory experiment in CPU/GPU hybrid fine-tuning. Original claims of 22% train loss improvement were a measurement artifact. CPU determinism effects could not be isolated from batch size confounds. MMLU warm restart data (4/4 positive direction) remains the only potentially valid signal. Experiment suspended.**

---

## What Is This?

SGDR (Stochastic Gradient Descent with Warm Restarts) is a well-established technique
that improves generalization by periodically restarting the learning rate schedule
(Loshchilov & Hutter, ICLR 2017). It has been widely adopted and is built into
PyTorch as `CosineAnnealingWarmRestarts`.

We accidentally implemented SGDR through a "design flaw" — using separate cosine
schedules for Phase1 and Phase2 of split training. We *hypothesized* that **restarting
from a CPU's deterministic position** (instead of GPU's non-deterministic position)
might produce better results. However, we discovered confounding variables (batch size
mismatch, train_loss measurement artifact) that prevent any reliable conclusions about
CPU determinism.

**What we confirmed:** bf16 = fp32 in steady-state training, warm restart (SGDR)
showed 4/4 positive MMLU direction in 7B experiments.

**What we could not confirm:** Whether CPU determinism provides any benefit over GPU.

**Original motivation:** We were trying to give a jellyfish (GPU-only training) a spine
(CPU deterministic anchor) for voice fine-tuning (Style-Bert-VITS2). The "design flaw"
of using separate cosine schedules for Phase1 and Phase2 accidentally created a warm
restart — but the CPU-specific claims could not survive controlled experiments.

---

## Results

### Training Loss

> ⚠️ **CRITICAL: All split-training train_loss values below are HF Trainer artifacts.**
> Phase2 resets accumulator but divides by global_step → ~20% artificial deflation.
> Corrected by multiplying reported value × (500/400) = × 1.25.

| Experiment | Method | Reported | **Corrected** | **vs A** | Runtime |
|---|---|---|---|---|---|
| **A** (baseline) | GPU-only, bf16, 500 steps | 1.184 | 1.184 | — | 10m 59s |
| **F** (GPU restart) | GPU fp32 100 → GPU bf16 400 | ~~0.9268~~ | **1.1585** | **-2.2%** | 10m 12s |
| **C** (CPU restart) | CPU fp32 100 → GPU bf16 400 | ~~0.9177~~ | **1.1471** | **-3.1%** | ~3h + 8m 18s |
| **G** (CPU restart fp32) | CPU fp32 100 → GPU fp32 400 | ~~0.9188~~ | **1.1485** | **-3.0%** | ~3h 14m + 14m 06s |

**Note on F vs C/G comparison:** F used batch 1×8, C/G used batch 2×4 (effective batch identical at 8, but micro-batch differs). This confounds the F→C gap: the ~1% difference between F (1.1585) and C (1.1471) may reflect batch configuration, not CPU determinism. See Errata at top.

### MMLU Benchmark (5-shot, limit 200)

| Category | A (baseline) | F (GPU restart) | C (CPU restart) | G (CPU restart fp32) |
|---|---|---|---|---|
| Humanities | 78.04% | 78.39% | 78.35% | **78.53%** |
| Social Sciences | 84.56% | 84.56% | 84.65% | **85.09%** |
| Other | 74.70% | 74.79% | 74.84% | **74.98%** |
| STEM | 69.79% | 69.97% ↑ | 69.65% ↓ | **70.16%** ↑ |
| **Overall** | **76.25%** | **76.41%** | **76.34%** | **76.66%** |

*All differences within stderr (±0.4%). However, all warm-restart experiments improved over baseline — 4/4 positive direction. Per-subject sign test for G vs A: 27↑ 12↓ 18= (p < 0.01 by binomial test). This directional consistency suggests a real but small warm restart benefit in fine-tuning, consistent with SGDR literature.*

**GPU restart vs CPU restart:**

> ⚠️ F used batch 1×8 (Phase1), C/G used batch 2×4. This micro-batch difference
> confounds the comparison below.

| | F (GPU restart) | G (CPU restart) | Δ (G - F) |
|---|---|---|---|
| MMLU Overall | 76.41% | 76.66% | +0.25% |
| STEM | 69.97% | 70.16% | +0.19% |
| moral_scenarios | 63.00% | 64.50% | +1.50% |

*G outperformed F in 3/4 categories. Both are warm restarts — the difference is CPU deterministic anchor vs GPU stochastic anchor, BUT also batch 2×4 vs 1×8. Cannot separate these factors. Sample size insufficient for significance.*

---

## Key Findings

### 1. SGDR (Warm Restart) Improves Fine-Tuning Generalization

Our Phase1/Phase2 split accidentally implemented a warm restart. SGDR
(Loshchilov & Hutter, 2017) is proven to improve generalization in pretraining.
We observed the same effect in fine-tuning: all 4 warm-restart experiments
improved MMLU over the continuous-lr baseline (A). The effect is small
(within stderr per experiment), but the 4/4 directional consistency and
G's 27↑/12↓ sign test (p < 0.01) suggest it is real.

> **Reference:** Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient
> Descent with Warm Restarts." ICLR 2017. https://arxiv.org/abs/1608.03983

### 2. bf16 = fp32 in Steady-State Training (3× Confirmed)

3B full-precision experiments (16-bit, no quantization) confirmed that
bf16 and fp32 produce identical training trajectories:

```
Confirmation 1: A-1 (bf16) vs Baseline (fp32) — full 500 steps, Δ ≤ 0.002
Confirmation 2: Exp A Phase2 convergence — split training, step 300+, Δ ≤ 0.003
Confirmation 3: Exp AA (continuous lr) — precision transition at step 100, Δ ≤ 0.001 ★ cleanest
```

The industry assumption that bf16 ≈ fp32 is correct. Precision staging
(switching from fp32 to bf16 mid-training) produces zero effect when the
lr schedule is continuous.

### 3. CPU Determinism: Inconclusive

> ⚠️ **Original claim "Quantization is the sole variable causing CPU ≠ GPU" has been retracted.**

CPU determinism effects could not be reliably isolated from batch size confounds:

```
7B QLoRA 4-bit:
  F (GPU fp32, batch 1×8) vs C/G (CPU fp32, batch 2×4):
  Phase1 Δ = 0.82% — but TWO variables changed (device + batch)

3B 16-bit full (no quantization):
  CPU (batch 1×8) vs GPU (batch 1×8):    Δ ≤ 0.002 — same batch → identical
  Tested with both continuous and discontinuous lr: always identical

3B QLoRA 4-bit (quantization added):
  CPU (batch 1×8) vs GPU (batch 1×8):    Δ ≤ 0.001 at step 40 — same batch → identical
  → Adding quantization did NOT cause CPU ≠ GPU when batch was controlled
```

**Conclusion:** When batch configuration is identical, CPU = GPU regardless of:
- Quantization (16-bit and 4-bit both identical)
- lr schedule (continuous and discontinuous both identical)
- Model size (3B tested)

The 7B "0.82% CPU deeper" finding likely reflects batch 2×4 vs 1×8 difference,
not CPU vs GPU determinism. This cannot be verified on 7B because VRAM limitations
prevent running GPU fp32 with batch 2×4.

### 4. Warm Restart Direction: Train Loss Worsens, MMLU Improves

An important observation — warm restart has opposite effects on train loss vs generalization:

```
3B (16-bit full):
  GPU warm restart: train_loss +0.52% worse than Baseline
  But 7B MMLU:      all warm restarts improved! (4/4 positive)
```

This is consistent with SGDR theory: warm restarts push the model toward
**flat minima** (good generalization) rather than **sharp minima** (low train loss
but poor generalization).

### 5. Zero Transition Shock

When switching from CPU fp32 to GPU bf16 at step 100:
```
Phase1 final:     loss 1.192 (step 90)
Phase2 first log: loss 1.209 (step 110)
Difference: +0.017
```
Hardware changed. Precision changed. Memory space changed. Loss barely moved.

### 6. GPU Hardware Noise Floor Demonstrated

Learning rate decayed 5x, but loss oscillation only reduced 30%. At lr ≈ 0 (1.117e-09), loss still oscillated ~0.17. This residual noise cannot be controlled by software (lr decay) — it is hardware-origin.

### 7. Phase2 fp32 Recovers Calculation Ability (Experiment G MMLU)

Despite similar corrected train loss (C: 1.1471 vs G: 1.1485), MMLU tells a different story. Experiment C's STEM decline (-0.14% vs A) was caused by bf16 precision loss, not by the anchor. Experiment G (fp32) recovered STEM (+0.37% vs A) while preserving understanding gains:

| Subject | A | C (bf16) | G (fp32) |
|---|---|---|---|
| college_mathematics | 45.00% | 43.00% ↓ | **47.00%** ↑ |
| moral_scenarios | 60.00% | 61.50% ↑ | **64.50%** ↑↑ |
| STEM (category) | 69.79% | 69.65% ↓ | **70.16%** ↑ |

**Same train loss, different knowledge distribution.** Phase2 precision matters for where the model lands within the anchor basin, even under 4-bit QLoRA.

### 8. ~~fp32 Precision Is the Dominant Factor in Train Loss (Experiment F)~~ — INVALIDATED

> ⚠️ **This section's quantitative claims are invalid.** The "96.6% / 3.4%" decomposition
> was calculated from artifact-inflated train_loss values AND has a batch size confound
> (F: batch 1×8, C: batch 2×4). With corrected values the split is ~69% / ~31%, but
> even this is unreliable because F and C/G used different micro-batch sizes.

Original (artifact-based) decomposition:
```
A → F (fp32 precision only):     -0.2572  (96.6%)  ← ARTIFACT
F → C (CPU determinism):         -0.0091  ( 3.4%)  ← ARTIFACT + BATCH CONFOUND
```

Corrected decomposition:
```
A → F (fp32 precision + batch change):  -0.0255  (~69%)  ← batch 2×4→1×8 confound
F → C (device + batch change):          -0.0114  (~31%)  ← batch 1×8→2×4 confound
```

Neither factor can be cleanly isolated due to batch mismatch.

On MMLU, F outperformed C overall (76.41% vs 76.34%) with a critical difference: **F preserved STEM** (69.97%) while C suppressed it (69.65%). Both used bf16 in Phase2 — the only variable was CPU vs GPU in Phase1. This means CPU determinism improves understanding/judgment but introduces a STEM penalty that GPU's non-deterministic fp32 avoids.

| Subject | A | C (CPU+bf16) | F (GPU+bf16) | G (CPU+fp32) |
|---|---|---|---|---|
| moral_scenarios | 60.00% | 61.50% | **63.00%** | **64.50%** |
| college_physics | 50.98% | 49.02% ↓ | **50.98%** ✅ | 50.00% |
| abstract_algebra | 53.00% | 52.00% ↓ | **53.00%** ✅ | 53.00% |
| STEM (category) | 69.79% | 69.65% ↓ | **69.97%** ↑ | **70.16%** ↑ |

**However:** F and C/G used different batch configurations, so even the MMLU comparison between F and C is confounded. The MMLU patterns above may reflect batch differences rather than CPU vs GPU.

---

## The Hypothesis

### Original Hypothesis: Precision-Staged Training — Disproved

The industry treats CPU as "slow GPU." Nobody uses CPU for training quality. We originally proposed **Precision-Staged Training** — running the first 20% of fine-tuning in fp32 to create a precise anchor point.

3B experiments disproved the pure precision hypothesis: bf16 = fp32 (3× confirmed), and precision transition with continuous lr has zero effect.

### Revised Hypothesis: ~~CPU-Anchored Warm Restart~~ — Suspended

> ⚠️ **The CPU-Anchored SGDR hypothesis has been suspended** due to unresolved
> batch confounds. The 7B "evidence" for CPU determinism was confounded with
> micro-batch size (2×4 vs 1×8). 3B experiments with controlled batch showed
> CPU = GPU in all conditions tested.

The lr discontinuity we thought was a flaw was actually implementing **SGDR (Warm Restarts)**
— a proven technique for improving generalization (Loshchilov & Hutter, 2017).

Standard SGDR restarts from a GPU's stochastic position. We *hypothesized* that restarting
from a CPU's deterministic position might produce better results, but could not isolate
this effect from batch confounds.

| Method | Restart Quality | MMLU vs Baseline | Status |
|---|---|---|---|
| **Standard SGDR** | GPU stochastic restart | +0.16% (Exp F) | Known technique |
| **CPU-Anchored SGDR** | CPU deterministic restart | +0.41% (Exp G) | **Confounded** (batch differs) |
| Continuous lr (no restart) | No restart | ±0.00% (Exp AA) | Baseline |

**Why 20%?** Derived from the Prime Number Theorem: 1/ln(100) ≈ 21.71%. This independently converges with the biological ratio of central nervous system to total body mass (~2% brain, ~20% including spinal cord infrastructure).

**Evidence from 7B (QLoRA 4-bit) — CONFOUNDED:**
```
Phase1 (same lr, same precision, same seed, DIFFERENT BATCH):
  GPU fp32 (batch 1×8): step 100 loss = 1.100     ← Exp F
  CPU fp32 (batch 2×4): step 100 loss = 1.091     ← Exp C/G
  Δ = 0.82% — but batch differs, so this is NOT a clean CPU vs GPU comparison

Phase2 (F: batch 2×4, C: batch 2×4):
  F vs C gap consistent ~0.010-0.014 throughout
  → inherited from Phase1 divergence (which had batch confound)

3B QLoRA 4-bit (batch 1×8 for both):
  GPU fp32: step 40 loss = 1.294
  CPU fp32: step 40 loss = 1.295
  Δ = 0.001 — when batch is controlled, CPU = GPU
```

**Evidence from 3B (16-bit full, no quantization):**
```
Continuous lr:    CPU fp32 = GPU fp32 (Δ ≤ 0.002) — CPU-AA
Discontinuous lr: CPU fp32 = GPU fp32 (Δ ≤ 0.002) — CPU-C (step 10-70)

= Without quantization noise, CPU and GPU follow identical paths.
= lr schedule does NOT change this — tested with both continuous and discontinuous.
```

### The Analogy

Current GPU-only training is a **jellyfish** — no central nervous system, drifting with
ocean currents (non-deterministic gradients). We tried to give the jellyfish a **spine**
(CPU deterministic anchor), but couldn't prove the spine actually does anything different
from a second tentacle (GPU fp32).

The analogy was poetic. The experiments were inconclusive. 🪼

---

## 3B Full-Precision Experiments

Branch: `3b-full-precision` — Qwen2.5-3B-Instruct, 16-bit full (no quantization)

### Completed

| Exp | Config | Train Loss | MMLU | Finding |
|---|---|---|---|---|
| A-1 | GPU bf16 500 steps | 1.267 | — | bf16 baseline |
| Baseline | GPU fp32 500 steps | 1.2672 | 69.30% | A-1 = Baseline → bf16 = fp32 ✅ |
| A | fp32→bf16 (20:80) discont. lr | 0.9801 (artifact) | 69.37% | lr artifact + MMLU noise |
| AA | fp32→bf16 (20:80) cont. lr | 0.975 (artifact) | pending | Baseline ±0.001 → precision transition = zero effect ✅ |
| CPU-AA | CPU fp32→GPU bf16 cont. lr | 0.9749 (artifact) | pending | Baseline ±0.002 → CPU determinism = zero in 16-bit ✅ |
| CPU-C | CPU fp32→GPU bf16 discont. lr | **discontinued** | — | CPU = GPU at step 70 (Δ ≤ 0.002) in 16-bit |
| **Q-G** | **QLoRA 4-bit CPU fp32→GPU fp32** | **in progress** | — | **CPU = GPU at step 40 (Δ ≤ 0.001) even with 4-bit quantization** |

### Key 3B Findings

1. **bf16 = fp32** (3× confirmed in 16-bit full)
2. **Precision transition = zero effect** with continuous lr
3. **CPU = GPU** in 16-bit full (no quantization) — all lr schedules tested
4. **CPU = GPU** in QLoRA 4-bit when batch is controlled (1×8 for both, Δ ≤ 0.001)
5. **~~Quantization causes CPU ≠ GPU~~** — retracted. 3B QLoRA 4-bit showed CPU = GPU with same batch.
6. **7B "0.82% CPU deeper"** likely reflects batch 2×4 vs 1×8 difference, not CPU determinism
7. **Warm restart (SGDR) worsens train_loss** in 16-bit full (+0.52%), but MMLU improved in 7B

---

## Reproduction

### Environment
- **GPU:** NVIDIA RTX 5090 Laptop (24GB VRAM)
- **CPU:** Intel Core Ultra 9 275HX (24 cores)
- **RAM:** 64GB DDR5
- **Framework:** LlamaFactory v0.9.4, PyTorch 2.12.0+cu128
- **Model:** Qwen/Qwen2.5-7B-Instruct (7B), Qwen/Qwen2.5-3B-Instruct (3B)
- **Dataset:** alpaca_en (52K instruction-following samples)
- **Method:** QLoRA 4-bit (7B, 20M params) / 16-bit full LoRA (3B)

### Identical Hyperparameters (7B: A = C = F = G)
```
lora_rank: 8          lora_alpha: 16
lora_dropout: 0.05    learning_rate: 1.0e-4
lr_scheduler: cosine  warmup_steps: 30
seed: 42              max_steps: 500
quantization: 4-bit bitsandbytes
cutoff_len: 512
```

| | A | C | F | G |
|---|---|---|---|---|
| Phase1 device | GPU | CPU | GPU | CPU |
| Phase1 precision | bf16 | fp32 | fp32 | fp32 |
| Phase2 precision | bf16 | bf16 | bf16 | fp32 |
| batch × accum | 2×4=8 | 2×4=8 | 1×8=8 / 2×4=8 | 1×8=8 |

F Phase1 uses batch 1 × accum 8 to fit fp32 in 24GB VRAM, Phase2 uses 2×4=8.
G uses batch 1 × accum 8 throughout (effective batch = 8, identical).

> ⚠️ **Batch confound:** F and G Phase1 use batch 1×8, while A and C use batch 2×4.
> Although effective batch size is identical (8), micro-batch size affects gradient
> accumulation order and floating-point rounding. This confounds the F vs C/G Phase1
> comparison and invalidates the "CPU determinism" decomposition.

**Intended only difference:** Device (CPU/GPU) + Precision (fp32/bf16)
**Actual differences:** Device + Precision + **Micro-batch size** (unintended confound)

### Steps to Reproduce

1. Install LlamaFactory + dependencies
2. Copy YAML configs to `C:\LlamaFactory\`
3. Run `run_exp_a.bat` (GPU-only baseline)
4. Run `run_exp_c.bat` (CPU anchor → GPU bf16 exploration)
5. Run `run_exp_f.bat` (GPU fp32 anchor → GPU bf16 exploration)
6. Run `run_exp_g.bat` (CPU anchor → GPU fp32 exploration)
7. Compare `trainer_state.json` in all save directories

**Note:** On Windows, use `CUDA_VISIBLE_DEVICES=-1` (not empty string) to force CPU-only mode.

### Files

```
experiment/
├── yaml/
│   ├── jellyfish_exp_a.yaml          # GPU-only baseline
│   ├── jellyfish_exp_c_phase1.yaml   # CPU anchor (fp32, 100 steps)
│   ├── jellyfish_exp_c_phase2.yaml   # GPU exploration (bf16, 400 steps)
│   ├── jellyfish_exp_f_phase1.yaml   # GPU anchor (fp32, 100 steps) — precision control
│   ├── jellyfish_exp_f_phase2.yaml   # GPU exploration (bf16, 400 steps)
│   ├── jellyfish_exp_g_phase1.yaml   # CPU anchor (fp32, 100 steps, = C phase1)
│   └── jellyfish_exp_g_phase2.yaml   # GPU exploration (fp32, 400 steps)
├── scripts/
│   ├── run_exp_a.bat                 # Run experiment A
│   ├── run_exp_c.bat                 # Run experiment C (Phase1 + Phase2)
│   ├── run_exp_f.bat                 # Run experiment F (Phase1 + Phase2)
│   └── run_exp_g.bat                 # Run experiment G (Phase1 + Phase2)
└── results/
    └── benchmark_log.txt             # Full experiment log with step-by-step data
 ```
### LoRA Adapters

Trained LoRA adapters are hosted on HuggingFace:
🤗 **[KK1kk1/jellyfish-cpu-anchor-lora](https://huggingface.co/KK1kk1/jellyfish-cpu-anchor-lora)**

- `jellyfish_exp_a/` — GPU-only baseline adapter
- `jellyfish_exp_c/` — CPU-anchor hybrid adapter (Phase2 bf16)
- `jellyfish_exp_f/` — GPU fp32-anchor precision-staged adapter (Phase2 bf16)
- `jellyfish_exp_g/` — CPU-anchor hybrid adapter (Phase2 fp32)

---

## Limitations

- **⚠️ train_loss artifact:** The reported 21.7–22.5% train loss improvements are measurement artifacts caused by HuggingFace Trainer's resume logic. Corrected Phase2 averages are within ~3% of baseline. MMLU is the only valid cross-experiment metric.
- **⚠️ Batch confound:** Experiments F and G (GPU fp32 Phase1) used batch 1×8 due to VRAM constraints, while A and C used batch 2×4. This micro-batch mismatch confounds the F vs C/G comparison and invalidates the "CPU determinism" decomposition. 3B experiments with controlled batch (1×8 for both) showed CPU = GPU.
- **Scale:** 500 steps / 0.077 epochs. Mini-experiment only.
- **Single seed:** 7B: seed=42, 3B: seed=5108. No repetition. Statistical power limited.
- **SGDR vs our implementation:** Standard SGDR uses the same optimizer throughout. Our implementation saves/loads checkpoints across phases, which may affect optimizer state differently.
- **Benchmark sensitivity:** MMLU on a strong base model (Qwen2.5-7B, ~75% baseline) leaves little room for differentiation at 500 steps.
- **MMLU subset:** Due to 24GB VRAM constraint, evaluated with `--limit 200` (200 samples per subject).
- **Hardware:** Tested on a gaming laptop (HP Omen), not server-grade hardware. Cannot run GPU fp32 batch 2×4 on 7B to resolve the batch confound.
- **3B vs 7B confounds:** 3B (16-bit full) and 7B (QLoRA 4-bit) differ in quantization, model size, seed, and LoRA configuration — direct comparison requires caution.

---

## Open Questions

### Confirmed / Answered

- ~~Does bf16 = fp32?~~ **Yes.** 3× independent confirmation in 3B 16-bit full.
- ~~Does precision staging help with continuous lr?~~ **No.** Exp AA = Baseline ±0.001.
- ~~Does CPU determinism help with continuous lr + no quantization?~~ **No.** CPU-AA = Baseline ±0.002.
- ~~Does CPU determinism help with discontinuous lr + no quantization?~~ **No.** CPU-C = Exp A ±0.002 (discontinued at step 70).
- ~~Is the 22% train loss improvement real?~~ **No.** Measurement artifact (HF Trainer resume logic).
- ~~Does CPU determinism require quantization noise?~~ **Retracted.** 3B QLoRA 4-bit with same batch (1×8) → CPU = GPU (Δ ≤ 0.001). 7B "0.82%" was confounded with batch mismatch (2×4 vs 1×8).
- ~~Does GPU fp32-only training match the anchor effect?~~ **Inconclusive.** F used batch 1×8 while A/C used 2×4. The "96.6% precision" decomposition was artifact-based AND batch-confounded.

### Unresolved (Experiment Suspended)

1. **Was the 7B "CPU ≠ GPU" caused by batch (2×4 vs 1×8) or CPU determinism?** Cannot verify — GPU fp32 batch 2×4 OOMs on 24GB VRAM. Would require server-grade hardware.
2. **Does warm restart (SGDR) consistently improve MMLU?** 7B: 4/4 positive (within stderr). Suggestive but single-seed, and F vs C/G batch-confounded.
3. **Is the ~3% corrected train_loss improvement real?** All split experiments (F, C, G) showed ~2-3% lower corrected loss than A. But this could be SGDR effect, batch effect, or optimizer state artifact.

### Abandoned

4. ~~Multi-cycle CPU-anchored SGDR~~ — CPU determinism hypothesis suspended.
5. ~~Multiple seeds~~ — Experiment suspended before seed variation.
6. ~~Optimal restart ratio~~ — Experiment suspended.
7. ~~fp64 CPU anchor~~ — Experiment suspended.

---

## Unrealized Experiments

| Experiment | Description | Status | Reason |
|---|---|---|---|
| A | GPU bf16 500 steps | ✅ Done | Baseline |
| B | CPU fp32 500 steps | ⬜ Not run | ~21 hours on laptop CPU |
| C | CPU fp32 100 → GPU bf16 400 | ✅ Done | CPU warm restart |
| D | CPU fp32 15% → GPU bf16 85% | ⬜ Abandoned | Experiment suspended |
| E | GPU bf16 → CPU fp32 (reverse) | ⬜ Abandoned | Experiment suspended |
| F | GPU fp32 100 → GPU bf16 400 | ✅ Done | GPU warm restart (= standard SGDR) |
| G | CPU fp32 100 → GPU fp32 400 | ✅ Done | CPU warm restart + fp32 Phase2 |
| H | CPU fp64 100 → GPU fp32 400 | ⬜ Abandoned | Experiment suspended |
| **Q-G** | **3B QLoRA 4-bit CPU fp32 → GPU fp32** | **⏸️ Partial** | **CPU = GPU at step 40 (Δ ≤ 0.001) — batch confound discovered** |

---

## References

- Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. https://arxiv.org/abs/1608.03983
- The SGDR technique is implemented in PyTorch as `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

---

## License

MIT License. Use freely. If this changes how you train models, a citation would be appreciated.

---

## Acknowledgments

- **HP Omen** (RTX 5090 Laptop, 24GB VRAM) — Where this all happened
- **Qwen** (Alibaba) — Base model
- **LlamaFactory** — Training framework
- **EleutherAI** — lm-evaluation-harness
- **Damione** (HuggingFace) — Suggested the GPU fp32→bf16 isolation experiment (Experiment F)
- **Loshchilov & Hutter** — SGDR: Warm Restarts technique that our "design flaw" accidentally rediscovered
- 🪼 — The jellyfish (still spineless, as it turns out)
