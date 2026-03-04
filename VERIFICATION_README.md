# 🪼 Project Jellyfish — Phase 2: SGDR & CPU Anchor Verification

## Background

Phase 1 (completed) established:
- **CPU = GPU = bf16 = fp32** in train_loss (Δ ≤ 0.005, all conditions)
- **Train_loss 22% improvement was a HF Trainer measurement artifact**
- **Exp F divergence (Δ=0.029) was a YAML config error** (3 fields missing)
- **MMLU warm restart: 4/4 positive direction** (within stderr individually)

Phase 1 left one critical question unanswered:
**Is the consistent MMLU improvement under warm restart real, and does CPU play a role?**

---

## Verification Questions

### Q1. Does CPU→GPU warm restart improve MMLU over baseline?
- Compare: A (GPU 500 straight) vs C (CPU fp32 100 → GPU bf16 400)
- **PASS:** C ≥ A in overall MMLU
- **FAIL:** C < A AND understanding/judgment domains also decline → **immediately discard**
- **GRACE (Q1-1):** C < A overall BUT understanding/judgment domains rise
  → Skip to Q4: if G ≥ A → grace accepted, expand groups for re-verification
  → If G < A → **discard** (CPU anchor doesn't save it)
  → If understanding/judgment domains are below baseline from the start → **discard** (no grace)

### Q2. Under Q1, do understanding/judgment domains rise distinctly?
- Tracked domains: moral_scenarios, professional_psychology, jurisprudence,
  medical_genetics, world_religions, moral_disputes
- Pass: majority of tracked domains show consistent improvement across groups
- Implication: **CPU creates a qualitatively different anchor** (not just noise)

### Q3. Under Q1, does MMLU rise regardless of domain?
- Check: do ALL 4 categories (humanities, social_sciences, other, STEM) improve?
- Pass: overall MMLU rises without category-specific trade-offs
- Implication: **SGDR warm restart is the primary mechanism**

### Q4. Under Q2, does CPU→GPU fp32 amplify the effect?
- Compare: C (CPU→GPU bf16) vs G (CPU→GPU fp32)
- Pass: G > C in understanding/judgment AND other domains also rise
- Implication: **CPU anchor + fp32 Phase2 = optimal configuration**

---

## Experiment Design

### Conditions per Group

| Label | Config | Phase1 | Phase2 | Purpose |
|-------|--------|--------|--------|---------|
| **A** | GPU bf16, 500 steps straight | — | — | Baseline (no restart) |
| **C** | CPU fp32 100 → GPU bf16 400 | CPU fp32, 100-step cosine | GPU bf16, resume 500-step cosine | CPU anchor + SGDR |
| **G** | CPU fp32 100 → GPU fp32 400 | CPU fp32, 100-step cosine | GPU fp32, resume 500-step cosine | CPU anchor + SGDR + fp32 Phase2 |

### Fixed Parameters (ALL experiments, ALL groups)

```
Model:              Qwen/Qwen2.5-7B-Instruct
Quantization:       QLoRA 4-bit (bitsandbytes)
LoRA:               rank=8, alpha=16, dropout=0.05, target=all
Dataset:            alpaca_en
Cutoff:             512
Total steps:        500 (A: 500 straight, C/G: 100+400)
lr:                 1.0e-4
lr_scheduler:       cosine
Warmup:             30 steps
Eval:               MMLU 5-shot, batch_size=1, limit=200
```

### YAML Config (must include ALL of these)

```yaml
trust_remote_code: true
quantization_method: bitsandbytes
dataloader_num_workers: 0
```

⚠️ Phase 1 Exp F was missing these 3 fields, causing Δ=0.029 divergence.
All Phase 2 experiments MUST include them.

### Group Seeds

| Group | Seed | Status |
|-------|------|--------|
| Group 1 | 42 | ✅ Complete (A, C, G) |
| Group 2 | 1234 | ⬜ Pending |
| Group 3 | 5108 | ⬜ Pending |
| Group 4 | 7777 | ⬜ Conditional (if 1-3 all pass) |
| Group 5 | 2025 | ⬜ Conditional (if 1-4 all pass) |

### Batch Configuration

| Experiment | Phase1 batch | Phase2 batch |
|------------|-------------|-------------|
| A | — | 2×4=8 (bf16) |
| C | 2×4=8 (CPU fp32) | 2×4=8 (GPU bf16) |
| G | 2×4=8 (CPU fp32) | 1×8=8 (GPU fp32, VRAM limit) |

Note: Phase 1 confirmed batch 2×4 = 1×8 in train_loss.
G Phase2 uses 1×8 due to VRAM (7B × fp32 + optimizer = >24GB with batch=2).

---

## Pass/Fail Criteria

### Gate System

```
Group-level gate:
  Q1 FAIL (C < A + understanding/judgment also down) in ANY group → that group FAILS
  Q1 GRACE (C < A but understanding/judgment up) → check Q4:
    G ≥ A → grace accepted, expand groups for re-verification
    G < A → FAIL
  ALL groups must pass or grace Q1 to proceed to Q2-Q4 analysis

Escalation:
  Groups 1-3 complete:
    Any group fails Q1 → STOP. Conclusion: "noise"
    All 3 pass Q1 → expand to 5 groups
  Groups 1-5 complete:
    All 5 pass Q1 → expand to 7 groups (seeds 314, 9999)
    → then 11 groups (seeds 42, 1234, 5108, 7777, 2025, 314, 9999, 8888, 3141, 1111, 6502)
  11 groups complete → final conclusion

Statistical threshold:
  Q1: sign test across groups (C > A in majority)
  Q2: per-domain consistency across groups
  Q4: paired comparison C vs G across groups
```

---

## Hardware & Runtime

```
Machine:    HP Omen, RTX 5090 Laptop 24GB VRAM, i9-275HX, 64GB DDR5
OS:         Windows (LlamaFactory venv)
A runtime:  ~11 min (GPU only)
C runtime:  ~3h (CPU Phase1) + ~8 min (GPU Phase2) = ~3h 10min
G runtime:  ~3h (CPU Phase1) + ~14 min (GPU Phase2 fp32) = ~3h 15min
MMLU eval:  ~40 min per experiment

Per group:  A + C + G + 3× MMLU = ~8.5 hours
3 groups:   ~25.5 hours
5 groups:   ~42.5 hours
```

---

## Files

| File | Content |
|------|---------|
| `VERIFICATION_README.md` | This file — experiment design & criteria |
| `Test_Group/Group_1.md` | Seed 42 — complete data (A, C, G train loss + MMLU) |
| `Test_Group/Group_2.md` | Seed 1234 — template (to be filled) |
| `Test_Group/Group_3.md` | Seed 5108 — template (to be filled) |

---

## Current Status

- [x] Group 1 (seed 42): A, C, G complete with MMLU
- [ ] Group 2 (seed 1234): not started
- [ ] Group 3 (seed 5108): not started

---

*Project Jellyfish 🪼 — "Does the heartbeat (lr restart) make the jellyfish smarter?"*
*Phase 2: Multi-seed SGDR & CPU Anchor Verification*
