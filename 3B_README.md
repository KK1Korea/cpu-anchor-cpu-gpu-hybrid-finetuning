# Project Jellyfish 🪼 — 3B Full-Precision Branch

## Branch Purpose

Validate the CPU-Anchor effect discovered in 7B QLoRA 4-bit experiments **without the quantization bottleneck**.

> ⚠️ **EXPERIMENT CONCLUDED (2026-03-04)**
> CPU determinism hypothesis **disproved**. 7B direct evidence: Exp A (GPU bf16,
> batch 2×4) = C/G (CPU fp32, batch 2×4) at Δ ≤ 0.005. CPU = GPU on 7B too.
> The "0.82% CPU deeper" was entirely micro-batch confound (F: 1×8 vs C/G: 2×4).
> 3B QLoRA 4-bit (BF-8) independently confirmed CPU = GPU with controlled batch.

| Item | 7B (Completed) | 3B (Completed) |
|------|----------------|--------------|
| Model | Qwen2.5-7B-Instruct | Qwen2.5-3B-Instruct |
| Loading | 4-bit QLoRA | **16-bit full loading** / QLoRA 4-bit |
| VRAM | ~6GB (4-bit) | ~22GB (16-bit + optimizer) |
| Quantization noise | σ ≈ 0.03 | σ ≈ 0 (16-bit) / σ ≈ 0.03 (4-bit) |
| MMLU headroom | ~1.5% (near ceiling) | ~15%+ (large headroom) |
| CPU vs GPU (16-bit) | Not tested | **CPU = GPU (0.000%)** |
| CPU vs GPU (4-bit) | CPU ≠ GPU (0.82%) ⚠️ **batch confound** | **CPU = GPU (0.002%)** ← batch controlled |

### ~~Core Discovery: Quantization Is the Key Variable~~ — RETRACTED

~~3B experiments conclusively demonstrated:~~

1. **bf16 = fp32** in steady-state training (3× confirmed) ✅ STILL VALID
2. **Precision staging = zero effect** with continuous lr ✅ STILL VALID
3. **CPU = GPU** in 16-bit full, regardless of lr schedule ✅ STILL VALID
4. ~~★ CPU ≠ GPU only in 7B QLoRA 4-bit → Quantization is the sole variable~~ **RETRACTED**

**What actually happened:** 3B QLoRA 4-bit with batch 1×8 → CPU = GPU (Δ ≤ 0.002).
7B's "0.82%" compared GPU batch 1×8 vs CPU batch 2×4 — batch mismatch, not quantization.

---

## Hardware

- HP Omen RTX 5090 Laptop
- VRAM: 24GB GDDR7
- CPU: Intel i9-275HX
- RAM: 64GB DDR5

### VRAM Budget (3B 16-bit)

```
Model weights (3B × 2bytes):     ~6GB
Optimizer state (fp32):           ~12GB
Gradient + activation:            ~4GB
──────────────────────────────────────
Total:                            ~22GB / 24GB ← fits
```

---

## 7B Completed Experiments Summary (Reference)

> ⚠️ train_loss values are HF Trainer artifacts (~20% deflation). MMLU is the only valid metric.
> ⚠️ F used batch 1×8, A/C used batch 2×4 — confounds F vs C comparison.

| Exp | Config | Reported Loss | **Corrected Loss** | MMLU | Batch (Phase1) |
|-----|--------|-----------|------|------|------|
| A | GPU bf16 100% | 1.184 | 1.184 | 76.25% | 2×4=8 |
| C | CPU fp32→GPU bf16 (20:80) | ~~0.9177~~ | **1.1471** | 76.34% | 2×4=8 |
| F | GPU fp32→GPU bf16 (20:80) | ~~0.9268~~ | **1.1585** | 76.41% | **1×8=8** ⚠️ |
| G | CPU fp32→GPU fp32 (20:80) | ~~0.9188~~ | **1.1485** | 76.66% | 1×8=8 |

### SGDR Discovery (Revised from "Three-Factor Model")

The lr discontinuity originally thought to be a "design flaw" was actually implementing
SGDR (Warm Restarts, Loshchilov & Hutter 2017). All warm-restart experiments improved
MMLU over baseline (4/4 positive direction, p = 0.0625).

```
Factor 1: SGDR warm restart             — Consistently improves MMLU (F, C, G > A)
Factor 2: CPU determinism in QLoRA 4-bit — RETRACTED (batch confound, see BF-8)
Factor 3: Phase2 precision               — Invisible in train loss, visible in MMLU distribution
```

### Key Finding: Understanding/Judgment Subjects Rise in ALL Experiments

```
                        A(base)   C        F        G
moral_scenarios         60.00%   61.50↑   63.00↑   64.50↑
professional_psychology 77.00%   78.00↑   77.50↑   79.00↑
medical_genetics        83.00%   84.00↑   84.00↑   84.00↑
jurisprudence           80.56%   81.48↑   81.48↑   81.48↑
world_religions         87.13%   88.30↑   89.47↑   87.72↑
```

Any model that passes through warm restart — CPU or GPU — **understanding/judgment always rises.**

---

## 3B Experiment Design

### Round 0 — Pure Precision Baselines

| Exp | Config | Purpose | Status |
|-----|--------|---------|--------|
| A-1 | GPU bf16 500 steps 100% | Floor reference | ✅ Done (train_loss = 1.267) |
| Baseline | GPU fp32 500 steps 100% | Ceiling reference | ✅ Done (train_loss = 1.2672, MMLU = 69.30%) |

**Result: bf16 = fp32 confirmed.** Δ = 0.0002 across 500 steps. Max per-step diff ≤ 0.002.
There is no ceiling/floor gap — they are identical.

Estimated time: ~25 min each

### Round 1 — Transition + CPU Determinism Tests

| Exp | Config | Purpose | Status |
|-----|--------|---------|--------|
| A | fp32→bf16 (20:80) discont.lr | Reproduce warm restart (SGDR) | ✅ Done (MMLU 69.37%, Δ+0.07% = noise) |
| **AA** | **fp32→bf16 (20:80) cont.lr** | **Design fix: continuous 500-step cosine** | **✅ Done (loss = Baseline ±0.001, MMLU pending)** |
| **CPU-AA** | **CPU fp32→GPU bf16 cont.lr** | **CPU determinism + continuous lr** | **✅ Done (loss = Baseline ±0.002, MMLU pending)** |
| **CPU-C** | **CPU fp32→GPU bf16 discont.lr** | **CPU determinism + warm restart** | **★ Discontinued (Phase1 CPU=GPU, step 70)** |

⚠️ **train_loss Artifact Warning (applies to ALL Phase1/Phase2 split experiments):**

When training resumes from a checkpoint (Phase2), HuggingFace Trainer resets the
loss accumulator but divides by `global_step` (total steps from both phases).
This means Phase2-only loss is divided by the full 500 steps instead of the actual
400 steps trained in Phase2.

```
Reported:   Exp A train_loss = 0.9801  (-22.7% vs Baseline)
Corrected:  0.9801 × (500/400) = 1.2251  (Phase2 actual average)
Baseline:   step 101-500 average ≈ 1.22
Reality:    Exp A ≈ Baseline in Phase2. No improvement.
```

This artifact affects ALL split-training experiments (7B: Exp F/C/G, 3B: Exp A/C/D).
The 7B "22% improvement" (JF series) is likely the same artifact.

**Exp AA — Design Fix:** Exp A used independent 100-step cosine for Phase1 (lr drops to ~0),
causing lr discontinuity at Phase2 resume. AA fixed this by using 500-step cosine for Phase1
(manually stopped at step 100), so lr continues smoothly. Result: AA = Baseline ±0.001 across
all 40 Phase2 steps. Exp A's +0.036 transition shock was 100% caused by lr discontinuity.

**train_loss is NOT a valid comparison metric for split-training experiments.**
MMLU is the only reliable cross-experiment comparison axis.

### Round 2 — QLoRA 4-bit Quantization Test (Batch-Controlled)

> ⚠️ **This round was intended to confirm "quantization = sole variable" but DISPROVED it.**

| Exp | Config | Purpose | Status |
|-----|--------|---------|--------|
| Q-G GPU | 3B QLoRA 4-bit, GPU fp32, batch 1×8, 100 steps | GPU baseline under quantization | ✅ Done (train_loss = 1.3217) |
| Q-G CPU | 3B QLoRA 4-bit, CPU fp32, batch 1×8, 100 steps | CPU under quantization | ⏸️ Partial (50/100 steps) |

```
Phase1 comparison (step 10-50):
  ┌──────────┬──────────────┬──────────────┬──────────┐
  │ Step     │ GPU fp32     │ CPU fp32     │ Δ        │
  ├──────────┼──────────────┼──────────────┼──────────┤
  │ step  10 │ 1.806        │ 1.807        │ +0.001   │
  │ step  20 │ 1.590        │ 1.592        │ +0.002   │
  │ step  30 │ 1.387        │ 1.388        │ +0.001   │
  │ step  40 │ 1.294        │ 1.295        │ +0.001   │
  │ step  50 │ 1.358        │ 1.360        │ +0.002   │
  └──────────┴──────────────┴──────────────┴──────────┘

  Result: CPU = GPU (Δ ≤ 0.002) even with QLoRA 4-bit!
  → Quantization is NOT the variable.
  → 7B "0.82%" was batch confound (2×4 vs 1×8).
```

**Batch confound explanation:**
```
7B experiments (batch DIFFERENT):
  GPU fp32 (Exp F): batch 1×8 = 8  ← VRAM limitation
  CPU fp32 (Exp C/G): batch 2×4 = 8  ← no VRAM constraint
  Result: "CPU ≠ GPU" (Δ = 0.82%) → but TWO variables changed!

3B experiments (batch SAME):
  GPU fp32: batch 1×8 = 8
  CPU fp32: batch 1×8 = 8  ← CONTROLLED
  Result: CPU = GPU (Δ ≤ 0.002) → only device changed, no effect
```

### ★ CPU-C Discontinuation Decision (2026-03-04)

CPU-C was designed to test CPU determinism under warm restart (discontinuous lr).
Phase1 was discontinued at step 70/100 because:

```
         Exp A (GPU fp32)     CPU-C (CPU fp32)     Δ
step 10: 1.814               1.815               +0.001
step 20: 1.813               1.815               +0.002
step 30: 1.845               1.845                0.000
step 40: 1.558               1.557               -0.001
step 50: 1.488               1.487               -0.001
step 60: 1.400               1.400                0.000
step 70: 1.277               1.277                0.000
```

CPU = GPU (Δ ≤ 0.002) at every step, identical to CPU-AA finding.
Same anchor → same Phase2 → same MMLU. No scientific value in continuing.

### Interpretation Matrix

```
Round 0:
A-1 ≈ Baseline               → bf16 ≈ fp32 confirmed. ✅ CONFIRMED (Δ=0.0002)
A-1 >> Baseline               → bf16 underperforms fp32. ✗ Rejected.

Round 1 (MMLU-based only — train_loss comparison invalid due to artifact):
A(MMLU) > Baseline(MMLU)      → lr restart improves generalization. ✗ Not observed (+0.07%, noise)
A(MMLU) ≈ Baseline(MMLU)      → No transition benefit without quantization. ✅ CONFIRMED
AA(loss) ≈ Baseline(loss)     → Precision transition with continuous lr = zero effect. ✅ CONFIRMED (±0.001)
AA(MMLU) vs Baseline(MMLU)    → Final precision-only test. Pending (experiment suspended).
CPU-AA(loss) ≈ Baseline(loss) → CPU determinism with continuous lr = zero effect in 16-bit. ✅ CONFIRMED (±0.002)
CPU-AA(MMLU) vs Baseline(MMLU)→ CPU determinism MMLU test. Pending (experiment suspended).
CPU-C Phase1 = Exp A Phase1   → CPU = GPU under warm restart lr too. ✅ CONFIRMED (±0.002)
  ★ 3B 16-bit: CPU = GPU across ALL conditions tested (continuous, discontinuous)
  ★ 7B QLoRA 4-bit: CPU ≠ GPU (0.82%) — NOW KNOWN TO BE BATCH CONFOUND (see Round 2)

Round 2 (Quantization Test — DISPROVED hypothesis):
Q-G(4-bit): CPU = GPU (Δ≤0.002) → Quantization does NOT cause CPU ≠ GPU. ✅ CONFIRMED
  ★ When batch is controlled (1×8 for both), CPU = GPU in ALL conditions:
    16-bit continuous lr, 16-bit discontinuous lr, AND 4-bit discontinuous lr.
  ★ 7B "0.82%" was the ONLY case where batch differed (GPU 1×8 vs CPU 2×4).
  ★★★ "QUANTIZATION = sole variable" RETRACTED. Batch confound discovered. ★★★
```

### Round 3 — Long-Context Benchmark (Abandoned)

**Purpose:** Distinguish whether warm restart is "a trick for short problems" or "fundamental capability improvement"

| Benchmark | Tests | Context Length |
|-----------|-------|---------------|
| RULER NIAH variant | Information retrieval accuracy in long text | 4K / 8K / 16K / 32K |
| Multi-hop QA | Reasoning by connecting distant information | 8K / 16K |
| Document summarization (optional) | Key extraction + coherence maintenance | 16K+ |

```
Hypothesis A: "Anchor" = Fundamental model comprehension improvement
  → MMLU ↑ AND Long-context ↑
  → "Strengthened central nervous system → all behaviors improve"
  → Claim: "CPU-anchored SGDR fundamentally improves model capability"

Hypothesis B: "State transition" = Training trick (regularization)
  → MMLU ↑ BUT Long-context = No change
  → "Synapse connections shifted, fundamentals unchanged"
  → Claim scope narrowed: domain-specific optimization only
```

**Test targets:** Baseline vs G (strongest anchor: CPU fp32→GPU fp32)

### Evaluation Framework (3 Axes)

```
Axis 1: Train Loss     — Optimization efficiency (existing)
Axis 2: MMLU           — Knowledge/comprehension (existing)
Axis 3: Long-Context   — Context maintenance capability (future)

All 3 axes improve  → Paper-title-level claim
Only 2 axes improve → Effect scope limited
Only 1 axis improves → Training trick level
```

---

## Key Results Summary

### 3B Complete Picture (16-bit + 4-bit)

```
┌──────────────────────┬──────────────┬──────────────────────┐
│                      │ Continuous lr │ Discontinuous lr     │
│                      │              │ (SGDR warm restart)   │
├──────────────────────┼──────────────┼──────────────────────┤
│ GPU fp32 (16-bit)    │ AA = Base ✅  │ A = +0.52% loss ✅   │
│                      │ (±0.001)     │ MMLU +0.07% (noise)  │
├──────────────────────┼──────────────┼──────────────────────┤
│ CPU fp32 (16-bit)    │ CPU-AA = Base│ CPU-C = Exp A ✅      │
│                      │ (±0.002)     │ (Phase1 Δ≤0.002,    │
│                      │              │  discontinued)        │
├──────────────────────┼──────────────┼──────────────────────┤
│ QLoRA 4-bit (disc.lr)│ Not tested   │ Q-G: CPU = GPU ★     │
│                      │              │ (Δ≤0.002, step 10-50)│
└──────────────────────┴──────────────┴──────────────────────┘

Finding: CPU = GPU in ALL conditions tested on 3B:
  - 16-bit continuous lr: ✅ (CPU-AA)
  - 16-bit discontinuous lr: ✅ (CPU-C)
  - 4-bit discontinuous lr: ✅ (Q-G) ★ NEW
  - No experiment produced CPU ≠ GPU when batch was controlled.
```

### Cross-Model Comparison: 3B vs 7B

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                        │ 3B (16-bit full) │ 3B (QLoRA 4-bit) │ 7B (QLoRA 4-bit) │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ bf16 = fp32?           │ Yes (3× confirmed)│ Not tested       │ Not tested       │
│ Precision staging?     │ Zero effect       │ Not tested       │ Artifact (22%)   │
│ CPU = GPU? (cont. lr)  │ Yes (±0.002)     │ Not tested       │ Not tested       │
│ CPU = GPU? (disc. lr)  │ Yes (±0.002)     │ Yes (±0.002) ★   │ "NO" (0.82%) ⚠️  │
│ CPU vs GPU batch       │ 1×8 = 1×8        │ 1×8 = 1×8        │ 1×8 ≠ 2×4 ⚠️    │
│ SGDR MMLU improvement? │ +0.07% (1 sample)│ Not tested       │ 4/4 positive     │
│ Quantization?          │ None             │ 4-bit QLoRA      │ 4-bit QLoRA      │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ KEY FINDING            │ CPU = GPU        │ CPU = GPU        │ Batch confound!  │
│                        │ (smooth landscape)│ (rough landscape)│ (2×4 vs 1×8)    │
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┘

★ 3B QLoRA 4-bit CPU = GPU DISPROVES "quantization causes CPU ≠ GPU"
★ 7B Exp A (GPU bf16, 2×4) = C/G (CPU fp32, 2×4) at Δ ≤ 0.005 — DIRECT 7B PROOF
★ 7B "0.82%" came from F's different batch (1×8) — ONLY F diverges
★ "Quantization is the sole variable" — RETRACTED → Batch is the sole variable
```

---

## Data Management

### 3B_benchmark_log.txt (Concise)

```
=== BF-2: 3B A-1 (GPU bf16 100%) ===
Date: 2026-03-03
Model: Qwen2.5-3B-Instruct, 16-bit full loading
Steps: 500, lr: 2e-5, seed: 5108
Train loss: 1.267
sub-1.0: Yes (0.9967, step 380)
MMLU: pending
→ Detailed data: data.xlsx Sheet "Train Loss"

=== BF-8: 3B Q-G (QLoRA 4-bit, CPU vs GPU, batch-controlled) ===
Date: 2026-03-04
Model: Qwen2.5-3B-Instruct, QLoRA 4-bit
Steps: 100 (GPU complete, CPU 50/100 at suspension)
lr: 1e-4, seed: 5108, batch: 1×8=8
GPU train_loss: 1.3217 (100 steps)
CPU vs GPU: Δ ≤ 0.002 (step 10-50) → CPU = GPU
★ RETRACTED: "Quantization causes CPU ≠ GPU"
```

### data.xlsx (All Data)

```
Sheet 1: Summary         — All experiments at a glance (7B + 3B combined)
Sheet 2: Train Loss      — Step-by-step loss + line charts
Sheet 3: MMLU Raw        — 57 subjects × per-experiment scores
Sheet 4: MMLU Compare    — 7B vs 3B cross-comparison + bar charts
Sheet 5: Transition      — A/C/D transition experiment results
Sheet 6: Category        — 4-category breakdown + radar charts
Sheet 7: Factor Analysis — 3-factor decomposition visualization
Sheet 8: Long-Context    — RULER/Multi-hop results + length-wise charts
```

---

## Estimated Timeline

| Phase | Content | Time | Status |
|-------|---------|------|--------|
| Model download | Qwen2.5-3B-Instruct | ~5 min | ✅ Done |
| A-1 | GPU bf16 100% 500 steps | ~25 min | ✅ Done (1.267) |
| Baseline | GPU fp32 100% 500 steps | ~23 min | ✅ Done (1.2672) |
| Exp A | fp32→bf16 discont. lr | ~19 min | ✅ Done (MMLU 69.37%) |
| Exp AA | fp32→bf16 cont. lr | ~19 min | ✅ Done (loss = Base ±0.001) |
| CPU-AA | CPU fp32→GPU bf16 cont. lr | ~7.5h + 18min | ✅ Done (loss = Base ±0.002) |
| CPU-C | CPU fp32→GPU bf16 discont. lr | ~2h (partial) | ★ Discontinued (CPU=GPU confirmed) |
| **Q-G GPU** | **QLoRA 4-bit GPU fp32 Phase1** | **4m 12s** | **✅ Done (1.3217)** |
| **Q-G CPU** | **QLoRA 4-bit CPU fp32 Phase1** | **~50/100 steps** | **⏸️ Suspended (CPU=GPU, Δ≤0.002)** |
| MMLU eval (AA, CPU-AA) | 57-subject evaluation | ~40 min each | Abandoned (experiment suspended) |

---

## Open Questions

### Answered

- ~~Does bf16 = fp32?~~ **Yes.** 3× confirmed (A-1 vs Baseline, Exp A convergence, Exp AA) + **7B direct** (A bf16 vs C/G fp32, Δ ≤ 0.005).
- ~~Does precision staging help?~~ **No.** With continuous lr, zero effect (±0.001).
- ~~Does CPU determinism help in 16-bit?~~ **No.** CPU = GPU in all conditions (±0.002).
- ~~Does lr schedule change CPU vs GPU?~~ **No.** Both continuous and discontinuous: CPU = GPU.
- ~~Does quantization create CPU ≠ GPU?~~ **No (disproved).** 3B QLoRA 4-bit: CPU = GPU (±0.002). 7B direct: A (GPU, 2×4) = C/G (CPU, 2×4) at Δ ≤ 0.005.
- ~~Was 7B "0.82%" batch or CPU determinism?~~ **Batch.** 7B Exp A directly matches C/G (same batch 2×4). Only F (batch 1×8) diverges.

### Unresolved (Experiment Concluded)

1. **Why does micro-batch size (2×4 vs 1×8) affect QLoRA 4-bit training?** Mechanism unclear. Not investigated further.
2. **Does warm restart (SGDR) consistently improve MMLU?** 7B: 4/4 positive. 3B: eval abandoned.

### Abandoned

3. ~~Multi-cycle CPU-anchored SGDR~~ — CPU determinism hypothesis disproved.
4. ~~Multiple seeds~~ — Experiment concluded before seed variation.
5. ~~fp64 CPU anchor~~ — Experiment concluded.
6. ~~Long-context benchmarks~~ — Experiment concluded.

---

## Project Structure

```
GitHub: cpu-anchor-cpu-gpu-hybrid-finetuning/
├── main branch (7B QLoRA 4-bit)
│   ├── README.md
│   ├── benchmark_log.txt (JF-1~JF-10)
│   ├── mmlu_subcategory_section.md
│   └── Phase3_Training_Hybrid_System.md
│
└── 3b-full-precision branch
    ├── 3B_README.md (this document)
    ├── 3B_benchmark_log.txt (BF-1~BF-8)
    └── data.xlsx
```

---

## Credits

- **Damione** (HuggingFace) — Suggested Experiment F (precision vs determinism isolation)
- **Loshchilov & Hutter** — SGDR: Warm Restarts (ICLR 2017) that our "design flaw" rediscovered

---

*Last updated: 2026-03-04 (experiment concluded — CPU determinism disproved)*
*Project Jellyfish 🪼 — CPU-Anchor Hybrid Fine-tuning Research (concluded)*
