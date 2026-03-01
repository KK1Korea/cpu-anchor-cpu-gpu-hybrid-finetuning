# 🪼 Jellyfish: CPU-Anchor Hybrid Training

**The first empirical evidence that CPU deterministic computation can anchor GPU fine-tuning, achieving 22.5% lower training loss with equivalent generalization.**

---

## What Is This?

The entire AI industry trains models exclusively on GPUs. Nobody has asked: *"What if the first 20% of fine-tuning was done on CPU?"*

This project demonstrates that CPU fp32 deterministic training creates a precise "anchor point" in parameter space, and subsequent GPU bf16 training explores within that anchor's basin — never escaping it.

**Key result:** With identical hyperparameters, identical data, and identical model, simply changing *where* the first 20% of training runs produces 22.5% lower training loss while maintaining equivalent MMLU benchmark performance.

---

## Results

### Training Loss

| Experiment | Method | Train Loss | Runtime |
|---|---|---|---|
| **A** (baseline) | GPU-only, bf16, 500 steps | 1.184 | 10m 59s |
| **C** (hybrid) | CPU fp32 100 steps → GPU bf16 400 steps | **0.9177** | ~3h + 8m 18s |
| | | **-22.5%** | |

### MMLU Benchmark (5-shot, limit 200)

| Category | A (GPU-only) | C (Hybrid) | Δ |
|---|---|---|---|
| Humanities | 78.04% | 78.35% | +0.31 |
| Social Sciences | 84.56% | 84.65% | +0.09 |
| Other | 74.70% | 74.84% | +0.14 |
| STEM | 69.79% | 69.65% | -0.14 |
| **Overall** | **76.25%** | **76.34%** | **+0.09** |

*Difference within stderr (±0.4%). Not statistically significant — but critically, the 22.5% train loss reduction did NOT cause overfitting.*

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

---

## The Hypothesis

The industry treats CPU as "slow GPU." Nobody uses CPU for training quality. We propose:

| Training Phase | Hardware | Precision | Role |
|---|---|---|---|
| Pre-training | GPU 100% | bf16 | Maximum exploration (jellyfish mode) |
| **Fine-tuning** | **CPU 20% → GPU 80%** | **fp32 → bf16** | **Anchor + Exploration** |
| Inference | GPU 100% | varies | Execute established pathways |

**Why 20%?** Derived from the Prime Number Theorem: 1/ln(100) ≈ 21.71%. This independently converges with the biological ratio of central nervous system to total body mass (~2% brain, ~20% including spinal cord infrastructure).

**Why CPU?**
- CPU fp32: Deterministic. Same input → same output. Always.
- GPU bf16: Non-deterministic. Same input → different output each time. (Parallel reduction order, cuDNN kernel selection, atomic operations)

The anchor requires **determinism + precision**. GPU fp32 provides precision but not determinism. Only CPU provides both.

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

### Identical Hyperparameters (A = C)
```
lora_rank: 8          lora_alpha: 16
lora_dropout: 0.05    batch_size: 2
gradient_accum: 4     learning_rate: 1.0e-4
lr_scheduler: cosine  warmup_steps: 30
seed: 42              max_steps: 500
quantization: 4-bit bitsandbytes
cutoff_len: 512
```

**Only difference:** Device (CPU/GPU) + Precision (fp32/bf16)

### Steps to Reproduce

1. Install LlamaFactory + dependencies
2. Copy YAML configs to `C:\LlamaFactory\`
3. Run `run_exp_a.bat` (GPU-only baseline)
4. Run `run_exp_c.bat` (CPU anchor → GPU exploration)
5. Compare `trainer_state.json` in both save directories

**Note:** On Windows, use `CUDA_VISIBLE_DEVICES=-1` (not empty string) to force CPU-only mode.

### Files

```
experiment/
├── yaml/
│   ├── jellyfish_exp_a.yaml          # GPU-only baseline
│   ├── jellyfish_exp_c_phase1.yaml   # CPU anchor (fp32, 100 steps)
│   └── jellyfish_exp_c_phase2.yaml   # GPU exploration (bf16, 400 steps)
├── scripts/
│   ├── run_exp_a.bat                 # Run experiment A
│   └── run_exp_c.bat                 # Run experiment C (Phase1 + Phase2)
└── results/
    └── benchmark_log.txt             # Full experiment log with step-by-step data
 ```
### LoRA Adapters

Trained LoRA adapters are hosted on HuggingFace:
🤗 **[KK1kk1/jellyfish-cpu-anchor-lora](https://huggingface.co/KK1kk1/jellyfish-cpu-anchor-lora)**

- `jellyfish_exp_a/` — GPU-only baseline adapter
- `jellyfish_exp_c/` — CPU-anchor hybrid adapter

---

## Limitations

- **Scale:** 500 steps / 0.077 epochs. Mini-experiment only.
- **Single seed:** seed=42, no repetition. Statistical power limited.
- **Benchmark sensitivity:** MMLU on a strong base model (Qwen2.5-7B, ~75% baseline) leaves little room for differentiation at 500 steps.
- **MMLU subset:** Due to 24GB VRAM constraint, evaluated with `--limit 200` (200 samples per subject).
- **Hardware:** Tested on a gaming laptop (HP Omen), not server-grade hardware.

---

## Open Questions

1. Does the MMLU gap emerge at full scale (1 epoch+, 6500+ steps)?
2. Does a weaker base model (3B) with more headroom show clearer differentiation?
3. Do alternative benchmarks (TruthfulQA, needle-in-a-haystack) capture what MMLU cannot?
4. Does high-quality data (LIMA, Orca) + CPU anchor amplify the effect?
5. Is the 22.5% train loss improvement reproducible across seeds?
6. Does CPU determinism hold on APU (unified memory) architectures?
7. Does the anchor provide overfitting resistance at full scale?
8. What is the optimal CPU:GPU ratio? Is 20% ideal, or can less suffice?
9. Does GPU fp32-only training (no CPU) match the anchor effect, or is CPU determinism essential? (Isolates precision vs. determinism)
10. Does CPU fp32 anchor + GPU fp32 exploration outperform CPU fp32 + GPU bf16? (Tests whether maintaining precision in Phase2 amplifies the anchor effect)


---

## Unrealized Experiments

| Experiment | Description | Status | Reason |
|---|---|---|---|
| A | GPU-only 500 steps | ✅ Done | Baseline |
| B | CPU-only 500 steps | ⬜ Not run | ~21 hours on laptop CPU |
| C | CPU 100 → GPU 400 | ✅ Done | Hybrid |
| D | CPU 15% → GPU 85% | ⬜ Not run | Equipment limitation |
| E | GPU → CPU (reverse) | ⬜ Not run | Equipment limitation |
| F | GPU fp32-only 500 steps | ⬜ Not run | Isolates precision vs. determinism |
| G | CPU fp32 100 → GPU fp32 400 | ⬜ Not run | Tests precision retention in Phase2 |

---

## The Analogy

Current GPU-only training is a **jellyfish** — no central nervous system, drifting with ocean currents (hardware noise). It reaches destinations, but cannot choose them.

CPU-anchor hybrid training gives the jellyfish a **spine** — a deterministic central pathway that GPU exploration orbits around. The anchor decides *where*. The GPU decides *how deeply*.

---

## License

MIT License. Use freely. If this changes how you train models, a citation would be appreciated.

---

## Acknowledgments

- **HP Omen** (RTX 5090 Laptop, 24GB VRAM) — World's first CPU-anchor experiment, on a gaming laptop
- **Qwen** (Alibaba) — Base model
- **LlamaFactory** — Training framework
- **EleutherAI** — lm-evaluation-harness
- 🪼 — The jellyfish
