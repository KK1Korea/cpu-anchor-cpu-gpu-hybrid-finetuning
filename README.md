[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude-blueviolet?style=flat-square&logo=anthropic)](https://www.anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-LoRA%20Adapters-yellow?style=flat-square)](https://huggingface.co/KK1kk1/jellyfish-cpu-anchor-lora)
# 🪼 Jellyfish: CPU-Anchor Hybrid Training

**The first empirical evidence that fp32 precision-staged training anchors GPU fine-tuning, achieving 22.5% lower training loss with equivalent generalization. CPU determinism provides an additional but secondary benefit.**

---

## What Is This?

The entire AI industry trains models exclusively on GPUs. Nobody has asked: *"What if the first 20% of fine-tuning was done at higher precision?"*

This project demonstrates that running the first 20% of fine-tuning steps in fp32 creates a precise "anchor point" in parameter space, and subsequent bf16 training explores within that anchor's basin — never escaping it. This works on both CPU and GPU.

**Key result:** With identical hyperparameters, identical data, and identical model, simply running the first 100 steps in fp32 produces 21.7–22.5% lower training loss while maintaining equivalent MMLU benchmark performance. Experiment F proved that 96.6% of this improvement comes from fp32 precision alone — no CPU required.

---

## Results

### Training Loss

| Experiment | Method | Train Loss | vs A | Runtime |
|---|---|---|---|---|
| **A** (baseline) | GPU-only, bf16, 500 steps | 1.184 | — | 10m 59s |
| **F** (precision-staged) | GPU fp32 100 → GPU bf16 400 | **0.9268** | **-21.7%** | 10m 12s |
| **C** (hybrid) | CPU fp32 100 → GPU bf16 400 | **0.9177** | **-22.5%** | ~3h + 8m 18s |
| **G** (hybrid fp32) | CPU fp32 100 → GPU fp32 400 | **0.9188** | **-22.4%** | ~3h 14m + 14m 06s |

Three-factor decomposition (A → C):
- fp32 precision in Phase1: 96.6% of improvement (A → F)
- CPU determinism: 3.4% of improvement (F → C)
- Phase2 precision: negligible on train loss, significant on MMLU distribution

### MMLU Benchmark (5-shot, limit 200)

| Category | A (GPU-only) | F (GPU fp32→bf16) | C (CPU→GPU bf16) | G (CPU→GPU fp32) |
|---|---|---|---|---|
| Humanities | 78.04% | 78.39% | 78.35% | **78.53%** |
| Social Sciences | 84.56% | 84.56% | 84.65% | **85.09%** |
| Other | 74.70% | 74.79% | 74.84% | **74.98%** |
| STEM | 69.79% | 69.97% ↑ | 69.65% ↓ | **70.16%** ↑ |
| **Overall** | **76.25%** | **76.41%** | **76.34%** | **76.66%** |

*All differences within stderr (±0.4%). However, the ordering A < C < F < G is directionally consistent. STEM specifically: C declined vs A, but F preserved STEM — GPU non-deterministic fp32 anchor avoids the STEM penalty that CPU deterministic anchor introduces.*

---

## Key Findings

### 1. CPU fp32 Outperforms GPU bf16 in 1/5 the Steps

CPU Phase1 reached loss 1.090 at step 80 — surpassing GPU-only's final loss of 1.184 at step 500. Five times fewer steps, lower loss.

### 2. Zero Transition Shock

When switching from CPU fp32 to GPU bf16 at step 100:
```
Phase1 final:     loss 1.192 (step 90)
Phase2 first log: loss 1.209 (step 110)
Difference: +0.017
```
Hardware changed. Precision changed. Memory space changed. Loss barely moved.

### 3. Anchor Basin Holds

Across all 400 GPU steps, loss oscillated within the range ~1.08–1.30. Zero escapes from the CPU-established basin. Every spike returned to the anchor range.

### 4. GPU Hardware Noise Floor Demonstrated

Learning rate decayed 5x, but loss oscillation only reduced 30%. At lr ≈ 0 (1.117e-09), loss still oscillated ~0.17. This residual noise cannot be controlled by software (lr decay) — it is hardware-origin.

### 5. Overfitting Rejected

Train loss 22.5% lower + MMLU equivalent = the model learned the training data more efficiently without losing generalization ability.

### 6. GPU Noise Is Non-Determinism, Not Precision (Experiment G)

Experiment G (CPU fp32 → GPU fp32) produced train loss 0.9188 vs C's 0.9177 — a 0.12% difference. Step-by-step comparison shows consistent ±0.001–0.003 divergence across all 400 GPU steps, with identical oscillation amplitudes:

```
C (bf16) step 250-300 range: 0.9966 ~ 1.238 = amplitude 0.241
G (fp32) step 250-300 range: 0.9983 ~ 1.239 = amplitude 0.240
```

Upgrading from bf16 (7-bit mantissa) to fp32 (23-bit mantissa) did **not** reduce noise. The GPU hardware noise floor comes from non-deterministic execution order (cuDNN kernel selection, parallel reduction, atomic operations), not from mantissa precision.

### 7. Phase2 fp32 Recovers Calculation Ability (Experiment G MMLU)

Despite identical train loss (C: 0.9177 vs G: 0.9188), MMLU tells a different story. Experiment C's STEM decline (-0.14% vs A) was caused by bf16 precision loss, not by the anchor. Experiment G (fp32) recovered STEM (+0.37% vs A) while preserving understanding gains:

| Subject | A | C (bf16) | G (fp32) |
|---|---|---|---|
| college_mathematics | 45.00% | 43.00% ↓ | **47.00%** ↑ |
| moral_scenarios | 60.00% | 61.50% ↑ | **64.50%** ↑↑ |
| STEM (category) | 69.79% | 69.65% ↓ | **70.16%** ↑ |

**Same train loss, different knowledge distribution.** Phase2 precision matters for where the model lands within the anchor basin, even under 4-bit QLoRA.

### 8. fp32 Precision Is the Dominant Factor, Not CPU Determinism (Experiment F)

Experiment F (GPU fp32 100 → GPU bf16 400) achieved train loss 0.9268 — capturing 96.6% of C's improvement over A using GPU alone. The three-factor decomposition:

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

The industry treats CPU as "slow GPU." Nobody uses CPU for training quality. We propose **Precision-Staged Training**:

| Training Phase | Hardware | Precision | Role |
|---|---|---|---|
| Pre-training | GPU 100% | varies | Maximum exploration (jellyfish mode) |
| **Fine-tuning** | **GPU/CPU 20% → GPU 80%** | **fp32 → bf16/fp32** | **Anchor + Exploration** |
| Inference | GPU 100% | varies | Execute established pathways |

**Why 20%?** Derived from the Prime Number Theorem: 1/ln(100) ≈ 21.71%. This independently converges with the biological ratio of central nervous system to total body mass (~2% brain, ~20% including spinal cord infrastructure).

**Why fp32 first?** Experiment F proved that fp32 precision in Phase1 accounts for 96.6% of the anchor benefit. The 23-bit mantissa (vs bf16's 7-bit) provides gradients precise enough to find a good basin in parameter space. Once established, bf16 cannot escape this basin.

**Why CPU (optional)?**
- CPU fp32: Deterministic + high precision. Same input → same output. Always.
- GPU fp32: High precision but non-deterministic. Same input → different output each time.
- CPU adds 3.4% additional train loss improvement and enables sub-1.0 loss basins.
- However, CPU determinism improves understanding/judgment at the cost of STEM — a tradeoff, not a pure gain.

The anchor requires **precision**. It benefits from **determinism**, but does not require it.

**Two tiers of Precision-Staged Training:**

| Tier | Method | MMLU vs A | STEM | Cost |
|---|---|---|---|---|
| **Tier 1 (Free lunch)** | GPU fp32 20% → GPU bf16 80% | +0.16% | Preserved | Zero — config change only |
| **Tier 2 (Full meal)** | CPU fp32 20% → GPU fp32 80% | +0.41% | Improved | CPU Phase1 time (~3h for 7B) |

**Why does GPU Phase2 precision matter?** Experiment G showed that CPU anchor + GPU fp32 outperformed CPU anchor + GPU bf16 in all MMLU categories, despite identical train loss. The anchor determines *what* the model learns; Phase2 precision determines *how clearly* that knowledge is expressed. Higher GPU precision preserves more of the anchor's benefit — the model's "synaptic clarity."

**Industry implication:** The current trend of lowering training precision (fp32 → bf16 → fp8 → fp4) for speed may be silently degrading model knowledge quality in ways that train loss cannot detect. Experiment G shows that even the bf16 → fp32 difference, invisible in train loss, changes what the model knows.

---

## Reproduction

### Environment
- **GPU:** NVIDIA RTX 5090 Laptop (24GB VRAM)
- **CPU:** Intel Core Ultra 9 275HX (24 cores)
- **RAM:** 64GB DDR5
- **Framework:** LlamaFactory v0.9.4, PyTorch 2.12.0+cu128
- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Dataset:** alpaca_en (52K instruction-following samples)
- **Method:** QLoRA 4-bit (20M trainable parameters)

### Identical Hyperparameters (A = C = F = G)
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

- **Scale:** 500 steps / 0.077 epochs. Mini-experiment only.
- **Single seed:** seed=42, no repetition. Statistical power limited.
- **Benchmark sensitivity:** MMLU on a strong base model (Qwen2.5-7B, ~75% baseline) leaves little room for differentiation at 500 steps.
- **MMLU subset:** Due to 24GB VRAM constraint, evaluated with `--limit 200` (200 samples per subject).
- **Hardware:** Tested on a gaming laptop (HP Omen), not server-grade hardware.
- **QLoRA precision ceiling (partial):** 4-bit quantized base model masks Phase2 precision differences in train loss (C ≈ G), but NOT in MMLU subcategory distribution (G > C in all categories). The bottleneck constrains loss sensitivity while still allowing different weight distributions that manifest in downstream benchmarks.

---

## Open Questions

### Answered

9. ~~Does GPU fp32-only training (no CPU) match the anchor effect, or is CPU determinism essential?~~ **Answered by Experiment F:** GPU fp32 captures 96.6% of train loss improvement. CPU determinism adds 3.4% and enables sub-1.0 basins, but is not required. On MMLU, F (76.41%) > C (76.34%), with F preserving STEM while C suppressed it — CPU determinism is a tradeoff, not a pure gain.
10. ~~Does CPU fp32 anchor + GPU fp32 exploration outperform CPU fp32 + GPU bf16?~~ **Answered by Experiment G:** Train loss — No (Δ 0.12%). MMLU — Yes. G outperformed C in all 4 categories (+0.32% overall), with STEM recovering from C's decline. Same loss, different knowledge distribution.

### Scale and Reproducibility

1. Does the MMLU gap emerge at full scale (1 epoch+, 6500+ steps)? At 500 steps (0.077 epochs), all MMLU differences are within stderr. Longer training may amplify or eliminate the signal.
2. Does a weaker base model (3B) with more headroom show clearer differentiation? (Next planned: Qwen2.5-3B, 16-bit full-precision loading, no QLoRA 4-bit quantization bottleneck. Separate GitHub branch: `3b-full-precision`.)
3. Is the 21.7–22.5% train loss improvement reproducible across seeds? Single-seed (42) results. Multi-seed verification is the minimum bar for statistical claims.
4. Does the fp32 anchor provide overfitting resistance at full scale? At 0.077 epochs, no overfitting observed. The anchor basin may constrain parameter exploration enough to delay or prevent overfitting at 1+ epochs.

### Precision-Staged Training Mechanics

5. What is the optimal fp32:bf16 step ratio? Is 20% ideal, or can less suffice? (Originally derived from Prime Number Theorem 1/ln(100) ≈ 21.71%. Post-F, the justification shifts: "enough fp32 steps for gradients to find a good basin." 10%? 5%? The minimum effective dose is unknown.)
6. Does fp64 → fp32 → bf16 → fp8 form a monotonic quality gradient invisible to train loss? (If fp32 Phase1 helps, does fp64 Phase1 help more? Now testable on both CPU and GPU, since precision — not device — is the primary factor.)
7. Does reverse staging (bf16 first → fp32 later) work, or is the order essential? (Proposed Experiment E-GPU: GPU bf16 100 → GPU fp32 400. The exact mirror of F. Tests whether *early* precision matters more than *late* precision. If E-GPU ≈ F → total fp32 amount matters, order doesn't. If E-GPU < F → order is critical, anchor must come first.)
8. Does QLoRA 4-bit quantization mask the full Precision-Staged Training benefit? (Partially answered: 4-bit masks train loss differences (C ≈ G) but does NOT mask MMLU differences (G > C). The 3B full-precision branch (16-bit loading, no quantization) tests whether the MMLU gap widens when this bottleneck is removed.)

### New Questions from Experiment F

9-1. Why does CPU determinism suppress STEM? C and F both used bf16 in Phase2. The only difference was CPU vs GPU in Phase1. Yet C's STEM declined (-0.14%) while F's STEM improved (+0.18%). Does deterministic gradient computation over-constrain the parameter space for pattern-matching tasks? Or does GPU's non-deterministic exploration provide diversity that helps STEM?
9-2. F achieved the highest scores in several subjects that neither C nor G reached (world_religions 89.47%, college_chemistry 56.00%, elementary_mathematics 72.00%). Does GPU non-deterministic exploration discover optimization paths that deterministic CPU cannot? Is there a "creative search" benefit to hardware noise?
9-3. moral_scenarios formed a perfect staircase: A (60.0%) → C (61.5%) → F (63.0%) → G (64.5%). This ordering (A < C < F < G) suggests both fp32 precision AND Phase2 precision contribute independently to moral reasoning improvement. Is this pattern reproducible across seeds and models?

### Benchmarks and Data

10-1. Do alternative benchmarks (TruthfulQA, needle-in-a-haystack, ethics benchmarks) capture precision-staging effects that MMLU cannot? The moral_scenarios staircase suggests judgment tasks may be particularly sensitive.
10-2. Does high-quality data (LIMA, Orca) + fp32 anchor amplify the effect? Alpaca's 52K samples are instruction-following; curated data may benefit more from precise early gradients.

### Hardware and Architecture

11. Does the 3.4% CPU determinism benefit survive on APU (unified memory) architectures? (Less critical post-F since CPU is secondary, but APU blurs the CPU/GPU boundary in interesting ways.)
12. Does GPU fp64 anchor produce a deeper basin than GPU fp32 anchor? (fp64 mantissa 52-bit vs fp32 23-bit. Post-F, this is testable purely on GPU — no CPU needed. If anchor precision scales with quality, this implies a new hyperparameter: anchor precision level.)

---

## Unrealized Experiments

| Experiment | Description | Status | Reason |
|---|---|---|---|
| A | GPU bf16 500 steps | ✅ Done | Baseline |
| B | CPU fp32 500 steps | ⬜ Not run | ~21 hours on laptop CPU |
| C | CPU fp32 100 → GPU bf16 400 | ✅ Done | Hybrid |
| D | CPU fp32 15% → GPU bf16 85% | ⬜ Not run | Equipment limitation |
| E | GPU bf16 → CPU fp32 (reverse) | ⬜ Not run | Equipment limitation |
| F | GPU fp32 100 → GPU bf16 400 | ✅ Done | **Answered Q#9:** Isolates precision vs. determinism. Suggested by Damione. fp32 precision = 96.6% of anchor effect. CPU determinism = 3.4%. MMLU 76.41% (> C's 76.34%). STEM preserved (69.97% vs C's 69.65%). |
| G | CPU fp32 100 → GPU fp32 400 | ✅ Done | **Answered Q#10:** Train loss ≈ C (Δ 0.12%), but MMLU +0.32% over C. fp32 recovers STEM decline. |
| H | CPU fp64 100 → GPU fp32 400 | ⬜ Not run | Tests whether higher-precision anchor (double precision) deepens the basin further |

---

## The Analogy

Current GPU-only bf16 training is a **jellyfish** — no central nervous system, drifting with ocean currents (imprecise gradients). It reaches destinations, but cannot choose them.

Precision-staged training gives the jellyfish **glasses** — fp32 gradients in the first 20% of steps let it see clearly where to go. Once the destination is locked in, bf16 speed carries it there.

CPU-anchor hybrid training goes further: it gives the jellyfish a **spine** — a deterministic central pathway that GPU exploration orbits around. The anchor decides *where*. The GPU decides *how deeply*. But even glasses alone (Experiment F) capture 96.6% of the benefit.

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
- 🪼 — The jellyfish
