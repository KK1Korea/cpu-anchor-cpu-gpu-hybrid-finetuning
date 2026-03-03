# Project Jellyfish 🪼 — 3B Full-Precision Branch

## Branch Purpose

Validate the CPU-Anchor effect discovered in 7B QLoRA 4-bit experiments **without the quantization bottleneck**.

| Item | 7B (Completed) | 3B (Planned) |
|------|----------------|--------------|
| Model | Qwen2.5-7B-Instruct | Qwen2.5-3B-Instruct |
| Loading | 4-bit QLoRA | **16-bit full loading** |
| VRAM | ~6GB (4-bit) | ~22GB (16-bit + optimizer) |
| Quantization noise | σ ≈ 0.03 | σ ≈ 0 (removed) |
| MMLU headroom | ~1.5% (near ceiling) | ~15%+ (large headroom) |
| C vs G gap prediction | 0.32% (measured) | **2~4%** (15x amplification predicted) |

### Three Core Questions

1. **Does the fp32 anchor effect amplify when quantization bottleneck is removed?** (0.32% → 2~4%)
2. **Does a weaker base model show clearer understanding/judgment improvements?**
3. **Is precision transition itself regularization, or does the anchor fundamentally improve capability?**

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

| Exp | Config | Train Loss | MMLU | Note |
|-----|--------|-----------|------|------|
| A | GPU bf16 100% | 1.184 | 76.25% | Baseline |
| C | CPU fp32→GPU bf16 (20:80) | 0.9177 (-22.5%) | 76.34% | Anchor+bf16 |
| F | GPU fp32→GPU bf16 (20:80) | 0.9268 (-21.7%) | 76.41% | Precision control |
| G | CPU fp32→GPU fp32 (20:80) | 0.9188 (-22.4%) | 76.66% | Anchor+fp32 |

### Three-Factor Model (Established in JF-9)

```
Factor 1: fp32 precision anchor (96.6%) — Dominant. Understanding/judgment improves across all experiments.
Factor 2: CPU determinism (3.4%)        — Enables sub-1.0 basins, STEM tradeoff.
Factor 3: Phase2 precision              — Invisible in train loss, visible in MMLU distribution.
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

Any model that passes through fp32 Phase1 — CPU or GPU, Phase2 bf16 or fp32 — **understanding/judgment always rises.**

---

## 3B Experiment Design

### Round 0 — Baseline

| Exp | Config | Purpose |
|-----|--------|---------|
| Baseline | GPU fp32 500 steps 100% | Ceiling reference |

Estimated time: ~20 min

### Round 1 — Precision Transition (Single transition, direction/position test)

| Exp | Config | Direction | Position | Purpose |
|-----|--------|-----------|----------|---------|
| A | fp32→bf16 (20:80) | Downward | Early | Reproduce F (already verified in 7B) |
| C | bf16→fp32 (20:80) | Upward | Early | Reverse transition effect |
| D | fp32→bf16→fp32 (20:40:40) | Down then up | Double | Recovery effect |

Estimated time: ~20 min each, ~60 min total

**B (fp32 80%→bf16 20%) — temporarily deferred:**
- 400 steps of accumulated optimizer state + Phase4 "edge of stability" + precision switch = catastrophic perturbation risk
- Analogy: Taking off glasses while walking a tightrope (vs. early transition = taking off glasses on flat ground)
- May revisit after Round 1 results clarify transition dynamics

### Interpretation Matrix

```
A < baseline, C ≈ baseline          → Only "early precision + transition" combo works. Order is everything.
A < baseline, C < baseline          → "Transition itself" is regularization. Direction irrelevant.
D > A                               → Late fp32 recovery effect exists.
D ≈ A                               → Late precision irrelevant. Only early phase matters.
D < A                               → Second transition interferes (inertia collision).
```

### Round 2 — Long-Context Benchmark (New Addition)

**Purpose:** Distinguish whether fp32 anchor is "a trick for short problems" or "fundamental capability improvement"

| Benchmark | Tests | Context Length |
|-----------|-------|---------------|
| RULER NIAH variant | Information retrieval accuracy in long text | 4K / 8K / 16K / 32K |
| Multi-hop QA | Reasoning by connecting distant information | 8K / 16K |
| Document summarization (optional) | Key extraction + coherence maintenance | 16K+ |

```
Hypothesis A: "Anchor" = Fundamental model comprehension improvement
  → MMLU ↑ AND Long-context ↑
  → "Strengthened central nervous system → all behaviors improve"
  → Claim: "fp32 anchor fundamentally improves model capability"

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
Axis 3: Long-Context   — Context maintenance capability (new)

All 3 axes improve  → Paper-title-level claim
Only 2 axes improve → Effect scope limited
Only 1 axis improves → Training trick level
```

---

## Data Management

### benchmark_log.txt (Concise)

```
=== JF-11: 3B Baseline (GPU fp32 100%) ===
Date: 2026-03-XX
Model: Qwen2.5-3B-Instruct, 16-bit full loading
Steps: 500, lr: 2e-5
Train loss: X.XXXX
MMLU: XX.XX%
Long-context RULER: XX.XX%
→ Detailed data: data.xlsx Sheet "JF-11"
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

| Phase | Content | Time |
|-------|---------|------|
| Model download | Qwen2.5-3B-Instruct | ~5 min |
| Baseline | GPU fp32 100% 500 steps | ~20 min |
| Round 1 (A, C, D) | 3 transition experiments | ~60 min |
| MMLU evaluation × 4 | 57-subject evaluation | ~40 min |
| Long-Context evaluation | RULER + Multi-hop | ~30 min |
| **Total** | | **~2.5 hours** |

---

## Open Questions

1. **fp64 anchor:** Does the loss landscape contain deeper basin structures invisible at fp32 resolution?
   - CPU fp64 Phase1 (~6 hours) → GPU fp32 Phase2 → ~21 hours total
   - 3B fp32 results will provide hints: if basins deepen → "more exists below", if not → "this is the floor"

2. **Precision Oscillation:** Does alternating fp32↔bf16 create cumulative regularization?
   - Zero prior work. Completely unexplored territory.

3. **Minimum anchor threshold:** How many fp32 steps are sufficient for anchoring?
   - Testable with 1%, 5%, 10%, 20% ratio experiments

4. **Anchor ratio ≈ improvement rate?** 20% fp32 → 21.7% train loss improvement. Coincidence or law?
   - Verifiable with 10%, 30% ratio experiments

---

## Project Structure

```
GitHub: cpu-anchor-cpu-gpu-hybrid-finetuning/
├── main branch (7B QLoRA 4-bit)
│   ├── README.md
│   ├── benchmark_log.txt (JF-1~JF-9)
│   ├── mmlu_subcategory_section.md
│   └── Phase3_Training_Hybrid_System.md
│
└── 3b-full-precision branch (new)
    ├── 3B_README.md (this document)
    ├── 3B_benchmark_log.txt (JF-11~)
    └── data.xlsx
```

---

## Credits

- **Damione** (HuggingFace) — Suggested Experiment F (precision vs determinism isolation) and inspired the transition experiments by questioning whether the effect is position-independent
- To be updated after 3B results are confirmed

---

*Last updated: 2026-03-03*
*Project Jellyfish 🪼 — CPU-Anchor Hybrid Fine-tuning Research*
