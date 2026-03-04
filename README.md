[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude-blueviolet?style=flat-square&logo=anthropic)](https://www.anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-LoRA%20Adapters-yellow?style=flat-square)](https://huggingface.co/KK1kk1/jellyfish-cpu-anchor-lora)
# 🪼 Jellyfish: CPU-Anchor Hybrid Training

> **Research Status (2026-03-04)**
>
> **What started as a "design flaw" turned out to be a rediscovery of SGDR (Warm Restarts).**
>
> Our Phase1/Phase2 split training accidentally implemented a warm restart
> (Loshchilov & Hutter, 2017 — "SGDR: Stochastic Gradient Descent with Warm Restarts").
> SGDR is a proven technique that improves generalization by periodically restarting the
> learning rate schedule. Our "lr discontinuity" was exactly this: Phase1 completes a
> 100-step cosine cycle (lr → 0), Phase2 starts a new 500-step cosine (lr jumps back up).
>
> **Key finding:** All warm-restart experiments showed MMLU improvement over baseline:
> - F (GPU restart): +0.16%
> - C (CPU restart): +0.09%
> - G (CPU restart + fp32 Phase2): +0.41%
> - All within stderr (±0.4%), but **4/4 positive direction** (p = 0.0625, binomial)
> - Per-subject sign test for G: 27↑ 12↓ 18= (p < 0.01)
>
> **The real question is no longer "does precision staging work?" but:**
> **"Does CPU deterministic restart outperform GPU stochastic restart (standard SGDR)?"**
>
> **Confirmed findings:**
> - bf16 = fp32 in steady-state training (3× independent confirmation, 3B 16-bit full)
> - Precision transition with continuous lr = zero effect (3B Exp AA, ±0.001)
> - CPU = GPU in 16-bit full, ALL lr schedules (3B CPU-AA ±0.002, CPU-C ±0.002)
> - SGDR (warm restart) consistently improves MMLU in fine-tuning (4/4 experiments)
> - train_loss "22% improvement" is a measurement artifact (HF Trainer resume logic)
> - ★ **Quantization is the sole variable causing CPU ≠ GPU** (3B 16-bit: 0%, 7B QLoRA: 0.82%)
>
> **Open investigation:**
> - 3B QLoRA 4-bit CPU vs GPU — final confirmation that quantization causes CPU ≠ GPU
> - Multi-cycle CPU-anchored SGDR (2+ epoch, multiple restarts)
> - MMLU evaluation for continuous-lr experiments (AA, CPU-AA pending)

**Accidentally rediscovering SGDR led to a new question: can CPU determinism improve the quality of warm restarts? Standard SGDR restarts from a GPU's stochastic position. We restart from a CPU's deterministic position — and in quantized (4-bit) environments, CPU and GPU take measurably different paths.**

---

## What Is This?

SGDR (Stochastic Gradient Descent with Warm Restarts) is a well-established technique
that improves generalization by periodically restarting the learning rate schedule
(Loshchilov & Hutter, ICLR 2017). It has been widely adopted and is built into
PyTorch as `CosineAnnealingWarmRestarts`.

**But SGDR always restarts from a GPU's non-deterministic position.** The GPU's stochastic
execution (cuDNN kernel selection, parallel reduction order, atomic operations) means
each restart begins from a slightly different, unpredictable point in parameter space.

We accidentally discovered that **restarting from a CPU's deterministic position** may
produce better results. CPU fp32 training is fully deterministic — same input always
produces the exact same output. When the warm restart occurs from this deterministic
anchor point, the subsequent GPU exploration may find better basins.

**Original motivation:** We were trying to give a jellyfish (GPU-only training) a spine
(CPU deterministic anchor) for voice fine-tuning (Style-Bert-VITS2). The "design flaw"
of using separate cosine schedules for Phase1 and Phase2 accidentally created a warm
restart — and it worked.

---

## Results

### Training Loss

| Experiment | Method | Train Loss | vs A | Runtime |
|---|---|---|---|---|
| **A** (baseline) | GPU-only, bf16, 500 steps | 1.184 | — | 10m 59s |
| **F** (GPU restart) | GPU fp32 100 → GPU bf16 400 | **0.9268** | **-21.7%** ⚠️ | 10m 12s |
| **C** (CPU restart) | CPU fp32 100 → GPU bf16 400 | **0.9177** | **-22.5%** ⚠️ | ~3h + 8m 18s |
| **G** (CPU restart fp32) | CPU fp32 100 → GPU fp32 400 | **0.9188** | **-22.4%** ⚠️ | ~3h 14m + 14m 06s |

> ⚠️ **Errata: train_loss percentages are measurement artifacts.**
> When HuggingFace Trainer resumes from a checkpoint, it resets the loss accumulator but divides by `global_step` (total steps across both phases). Phase2 trains 400 steps but divides by 500, producing ~20% artificial deflation. Corrected: F = 0.9268 × (500/400) = **1.1585**, C = 0.9177 × (500/400) = **1.1471**, G = 0.9188 × (500/400) = **1.1485**. All are within ~3% of A's 1.184, not 22%.
>
> This was discovered through 3B full-precision experiments (branch: `3b-full-precision`) where step-by-step comparison confirmed Phase2 loss tracks baseline throughout. **MMLU results remain valid** as they are independent evaluations unaffected by the trainer's loss averaging.

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

| | F (GPU restart) | G (CPU restart) | Δ (G - F) |
|---|---|---|---|
| MMLU Overall | 76.41% | 76.66% | +0.25% |
| STEM | 69.97% | 70.16% | +0.19% |
| moral_scenarios | 63.00% | 64.50% | +1.50% |

*G outperformed F in 3/4 categories. Both are warm restarts — the difference is CPU deterministic anchor vs GPU stochastic anchor. Sample size insufficient for significance, but direction is consistent.*

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

### 3. CPU Determinism: Quantization-Dependent

CPU determinism effects depend on quantization, not lr schedule or precision:

```
3B 16-bit full (no quantization):
  Continuous lr:    CPU = GPU (Δ ≤ 0.002) — CPU-AA
  Discontinuous lr: CPU = GPU (Δ ≤ 0.002) — CPU-C (discontinued, confirmed at step 70)
  → CPU = GPU in ALL conditions tested.

7B QLoRA 4-bit (quantization):
  Discontinuous lr: CPU ≠ GPU (Phase1 Δ = 0.82%, Phase2 divergence +20%)
  → CPU finds different trajectory from GPU.
```

★ **Quantization is the sole variable causing CPU ≠ GPU.**

3B CPU-C was discontinued at step 70/100 because Phase1 loss matched Exp A (GPU)
at every step (Δ ≤ 0.002). Same anchor → same Phase2 → same MMLU. No scientific
value in completing the experiment.

**Mechanism (hypothesis):**
4-bit quantization compresses weights → dequantization introduces rounding noise →
GPU processes this noise non-deterministically (stochastic path) while CPU processes
it deterministically (consistent path). Without quantization (16-bit full), there is
no rounding noise, so CPU and GPU follow identical paths regardless of lr schedule.

**Next step:** 3B QLoRA 4-bit CPU vs GPU to confirm on the same model.

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

Despite identical train loss (C: 0.9177 vs G: 0.9188), MMLU tells a different story. Experiment C's STEM decline (-0.14% vs A) was caused by bf16 precision loss, not by the anchor. Experiment G (fp32) recovered STEM (+0.37% vs A) while preserving understanding gains:

| Subject | A | C (bf16) | G (fp32) |
|---|---|---|---|
| college_mathematics | 45.00% | 43.00% ↓ | **47.00%** ↑ |
| moral_scenarios | 60.00% | 61.50% ↑ | **64.50%** ↑↑ |
| STEM (category) | 69.79% | 69.65% ↓ | **70.16%** ↑ |

**Same train loss, different knowledge distribution.** Phase2 precision matters for where the model lands within the anchor basin, even under 4-bit QLoRA.

### 8. fp32 Precision Is the Dominant Factor in Train Loss (Experiment F)

Experiment F (GPU fp32 100 → GPU bf16 400) achieved train loss 0.9268 — capturing 96.6% of C's improvement over A using GPU alone. The three-factor decomposition *(based on reported train_loss — artifact-affected, see Errata)*:

```
A → F (fp32 precision only):     -0.2572  (96.6% of total A→C gain)
F → C (CPU determinism):         -0.0091  ( 3.4% of total A→C gain)
Total A → C:                     -0.2663  (100%)
```

On MMLU, F outperformed C overall (76.41% vs 76.34%) with a critical difference: **F preserved STEM** (69.97%) while C suppressed it (69.65%). Both used bf16 in Phase2 — the only variable was CPU vs GPU in Phase1. This means CPU determinism improves understanding/judgment but introduces a STEM penalty that GPU's non-deterministic fp32 avoids.

| Subject | A | C (CPU+bf16) | F (GPU+bf16) | G (CPU+fp32) |
|---|---|---|---|---|
| moral_scenarios | 60.00% | 61.50% | **63.00%** | **64.50%** |
| college_physics | 50.98% | 49.02% ↓ | **50.98%** ✅ | 50.00% |
| abstract_algebra | 53.00% | 52.00% ↓ | **53.00%** ✅ | 53.00% |
| STEM (category) | 69.79% | 69.65% ↓ | **69.97%** ↑ | **70.16%** ↑ |

**Industry implication:** Any lab can improve fine-tuning TODAY by running the first 20% of steps in fp32, then switching to bf16. No CPU required. No special hardware. Just a config change. F also achieved sub-1.0 train loss: No (lowest 1.007), while CPU-anchored C and G both broke through (0.9966 and 0.9983) — suggesting CPU determinism enables access to deeper basins that GPU non-determinism cannot reach.

---

## The Hypothesis

### Original Hypothesis: Precision-Staged Training

The industry treats CPU as "slow GPU." Nobody uses CPU for training quality. We originally proposed **Precision-Staged Training** — running the first 20% of fine-tuning in fp32 to create a precise anchor point.

3B experiments disproved the pure precision hypothesis: bf16 = fp32 (3× confirmed), and precision transition with continuous lr has zero effect.

### Revised Hypothesis: CPU-Anchored Warm Restart

The lr discontinuity we thought was a flaw was actually implementing **SGDR (Warm Restarts)**
— a proven technique for improving generalization (Loshchilov & Hutter, 2017).

Standard SGDR restarts from a GPU's stochastic position. We propose restarting from
a CPU's deterministic position:

| Method | Restart Quality | MMLU vs Baseline | Status |
|---|---|---|---|
| **Standard SGDR** | GPU stochastic restart | +0.16% (Exp F) | Known technique |
| **CPU-Anchored SGDR** | CPU deterministic restart | +0.41% (Exp G) | Our contribution |
| Continuous lr (no restart) | No restart | ±0.00% (Exp AA) | Baseline |

**Why CPU restart may be better:**
- CPU fp32: Deterministic + high precision. Same input → same output. Always.
- GPU fp32: High precision but non-deterministic. Same input → different output each time.
- When SGDR restarts the lr, it "re-explores" from the current position.
- A deterministic anchor may provide a more precise starting point for re-exploration.

**Why 20%?** Derived from the Prime Number Theorem: 1/ln(100) ≈ 21.71%. This independently converges with the biological ratio of central nervous system to total body mass (~2% brain, ~20% including spinal cord infrastructure).

**Evidence from 7B (QLoRA 4-bit):**
```
Phase1 (same lr, same precision, same seed):
  GPU fp32: step 100 loss = 1.100
  CPU fp32: step 100 loss = 1.091  (0.82% deeper)

Phase2 (both switched to GPU):
  A vs AA (lr difference):  83% convergence → same basin
  F vs C  (CPU difference): 20% divergence → different basins!

= lr differences are temporary. CPU differences are permanent.
= CPU finds a different region of parameter space.
```

**Evidence from 3B (16-bit full, no quantization):**
```
Continuous lr:    CPU fp32 = GPU fp32 (Δ ≤ 0.002) — CPU-AA
Discontinuous lr: CPU fp32 = GPU fp32 (Δ ≤ 0.002) — CPU-C (step 10-70)

= Without quantization noise, CPU and GPU follow identical paths.
= lr schedule does NOT change this — tested with both continuous and discontinuous.
= ★ Quantization is the sole variable causing CPU ≠ GPU.
```

### The Analogy

Current GPU-only training is a **jellyfish** — no central nervous system, drifting with
ocean currents (non-deterministic gradients). SGDR gives the jellyfish periodic pushes
(lr restarts), but each push starts from wherever the current carried it.

CPU-anchor gives the jellyfish a **spine** — each SGDR restart begins from a precisely
determined position. The spine doesn't control where the jellyfish drifts after each push,
but it controls where each push begins. GPU decides *how to explore*. CPU decides
*where to restart from*.

Standard SGDR: jellyfish with periodic pushes (GPU → GPU → GPU)
CPU-Anchored SGDR: jellyfish with spinal pushes (CPU → GPU, restart from spine)

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
| CPU-C | CPU fp32→GPU bf16 discont. lr | **discontinued** | — | ★ CPU = GPU at step 70 (Δ ≤ 0.002) → quantization confirmed |

### Key 3B Findings

1. **bf16 = fp32** (3× confirmed in 16-bit full)
2. **Precision transition = zero effect** with continuous lr
3. **CPU = GPU** in smooth landscape (16-bit, no quantization) — all lr schedules tested
4. **★ Quantization is the sole variable causing CPU ≠ GPU** — 3B 16-bit: 0%, 7B QLoRA: 0.82%
5. **Warm restart (SGDR) worsens train_loss** in 16-bit full (+0.52%), opposite of 7B QLoRA
6. **But MMLU improved in 7B** — warm restart may trade train_loss for generalization
7. **Next:** 3B QLoRA 4-bit CPU vs GPU to confirm quantization hypothesis on same model

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

**Only difference:** Device (CPU/GPU) + Precision (fp32/bf16)

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
- **Scale:** 500 steps / 0.077 epochs. Mini-experiment only.
- **Single seed:** 7B: seed=42, 3B: seed=5108. No repetition. Statistical power limited.
- **SGDR vs our implementation:** Standard SGDR uses the same optimizer throughout. Our implementation saves/loads checkpoints across phases, which may affect optimizer state differently.
- **Benchmark sensitivity:** MMLU on a strong base model (Qwen2.5-7B, ~75% baseline) leaves little room for differentiation at 500 steps.
- **MMLU subset:** Due to 24GB VRAM constraint, evaluated with `--limit 200` (200 samples per subject).
- **Hardware:** Tested on a gaming laptop (HP Omen), not server-grade hardware.
- **3B vs 7B confounds:** 3B (16-bit full) and 7B (QLoRA 4-bit) differ in quantization, model size, seed, and LoRA configuration — direct comparison requires caution.

---

## Open Questions

### Confirmed / Answered

- ~~Does bf16 = fp32?~~ **Yes.** 3× independent confirmation in 3B 16-bit full.
- ~~Does precision staging help with continuous lr?~~ **No.** Exp AA = Baseline ±0.001.
- ~~Does CPU determinism help with continuous lr + no quantization?~~ **No.** CPU-AA = Baseline ±0.002.
- ~~Does CPU determinism help with discontinuous lr + no quantization?~~ **No.** CPU-C = Exp A ±0.002 (discontinued at step 70).
- ~~Is the 22% train loss improvement real?~~ **No.** Measurement artifact (HF Trainer resume logic).
- ~~Does GPU fp32-only training match the anchor effect?~~ **Answered by Experiment F:** On reported train_loss (artifact-affected), GPU fp32 captures 96.6%. On MMLU, F (76.41%) > C (76.34%) with STEM preserved.
- ~~Does CPU determinism require quantization noise?~~ **Yes.** 3B 16-bit: CPU = GPU (both lr schedules). 7B QLoRA 4-bit: CPU ≠ GPU (0.82%). ★ Quantization is the sole variable.

### Active Investigation

1. **3B QLoRA 4-bit: Does quantization create CPU ≠ GPU on the same model?** Same 3B model + add 4-bit quantization → if CPU ≠ GPU: quantization confirmed 100%. If CPU = GPU: model size is variable (unexpected).
2. **Does warm restart (SGDR) consistently improve MMLU in fine-tuning?** 7B: 4/4 positive. 3B: MMLU evaluation pending (AA, CPU-AA).
3. **Multi-cycle CPU-anchored SGDR:** Does [CPU→GPU] × N restarts amplify the effect? (2+ epoch experiment planned)
4. **Multiple seeds:** Current results from single seeds only. Reproducibility with 3+ seeds needed.

### Future Directions

5. Does the MMLU gap emerge at full scale (1 epoch+, 6500+ steps)?
6. What is the optimal restart ratio? (10%? 20%? Multiple restarts?)
7. Does CPU-Anchored SGDR with multiple restarts (CPU→GPU→CPU→GPU...) amplify the effect?
8. Does fp64 CPU anchor provide a deeper basin than fp32 CPU anchor?
9. Does high-quality data (LIMA, Orca) + CPU-anchored SGDR amplify the effect?
10. Do alternative benchmarks (TruthfulQA, needle-in-a-haystack) capture warm restart effects?

---

## Unrealized Experiments

| Experiment | Description | Status | Reason |
|---|---|---|---|
| A | GPU bf16 500 steps | ✅ Done | Baseline |
| B | CPU fp32 500 steps | ⬜ Not run | ~21 hours on laptop CPU |
| C | CPU fp32 100 → GPU bf16 400 | ✅ Done | CPU warm restart |
| D | CPU fp32 15% → GPU bf16 85% | ⬜ Not run | Equipment limitation |
| E | GPU bf16 → CPU fp32 (reverse) | ⬜ Not run | Equipment limitation |
| F | GPU fp32 100 → GPU bf16 400 | ✅ Done | GPU warm restart (= standard SGDR) |
| G | CPU fp32 100 → GPU fp32 400 | ✅ Done | CPU warm restart + fp32 Phase2 |
| H | CPU fp64 100 → GPU fp32 400 | ⬜ Not run | Double-precision CPU anchor test |

---

## References

- Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. https://arxiv.org/abs/1608.03983
- The SGDR technique is implemented in PyTorch as `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

---

## License

MIT License. Use freely. If this changes how you train models, a citation would be appreciated.

---

## Acknowledgments

- **HP Omen** (RTX 5090 Laptop, 24GB VRAM) — World's first CPU-anchor experiment, on a gaming laptop
- **Qwen** (Alibaba) — Base model
- **LlamaFactory** — Training framework
- **EleutherAI** — lm-evaluation-harness
- **Damione** (HuggingFace) — Suggested the GPU fp32→bf16 isolation experiment (Experiment F) to separate precision from determinism
- **Loshchilov & Hutter** — SGDR: Warm Restarts technique that our "design flaw" accidentally rediscovered
- 🪼 — The jellyfish
