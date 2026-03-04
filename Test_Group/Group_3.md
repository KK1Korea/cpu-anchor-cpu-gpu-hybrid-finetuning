# Group 3 — Seed 5108

## Configuration

| Parameter | Value |
|-----------|-------|
| **Seed** | 5108 |
| **Model** | Qwen/Qwen2.5-7B-Instruct |
| **Quantization** | QLoRA 4-bit (bitsandbytes) |
| **LoRA** | rank=8, alpha=16, dropout=0.05, target=all |
| **Dataset** | alpaca_en |
| **Cutoff** | 512 |
| **lr** | 1.0e-4, cosine, warmup=30 |
| **Total steps** | 500 |
| **Eval** | MMLU 5-shot, batch_size=1, limit=200 |
| **Hardware** | HP Omen, RTX 5090 Laptop 24GB, i9-275HX, 64GB DDR5 |
| **Date** | TBD |
| **Status** | ⬜ Not started |

### Per-Experiment Config

| | A (Baseline) | C (CPU→bf16) | G (CPU→fp32) |
|---|---|---|---|
| Phase1 device | GPU | CPU | CPU |
| Phase1 precision | bf16 | fp32 | fp32 |
| Phase1 steps | 500 (straight) | 100 | 100 |
| Phase1 lr schedule | 500-step cosine | 100-step cosine | 100-step cosine |
| Phase1 batch | 2×4=8 | 2×4=8 | 2×4=8 |
| Phase2 device | — | GPU | GPU |
| Phase2 precision | — | bf16 | fp32 |
| Phase2 steps | — | 400 | 400 |
| Phase2 batch | — | 2×4=8 | 1×8=8 |
| Runtime | — | — | — |

---

## Train Loss — Phase1 (Step 10–100)

| Step | A (GPU bf16) | C (CPU fp32) | G (CPU fp32) | Δ C-A | Δ G-A |
|------|-------------|-------------|-------------|-------|-------|
| 10 | — | — | — | — | — |
| 20 | — | — | — | — | — |
| 30 | — | — | — | — | — |
| 40 | — | — | — | — | — |
| 50 | — | — | — | — | — |
| 60 | — | — | — | — | — |
| 70 | — | — | — | — | — |
| 80 | — | — | — | — | — |
| 90 | — | — | — | — | — |
| 100 | — | — | — | — | — |

---

## MMLU Results — Category

| Category | A (Baseline) | C (CPU→bf16) | G (CPU→fp32) | Δ C-A | Δ G-A |
|----------|-------------|-------------|-------------|-------|-------|
| Humanities | — | — | — | — | — |
| Social Sciences | — | — | — | — | — |
| Other | — | — | — | — | — |
| STEM | — | — | — | — | — |
| **Overall** | — | — | — | — | — |

### Q1 Check: C > A?
⬜ **PENDING**

### Q2 Check: Understanding/Judgment domains rise?
| Domain | A | C | G | C-A | G-A |
|--------|---|---|---|-----|-----|
| moral_scenarios | — | — | — | — | — |
| professional_psychology | — | — | — | — | — |
| jurisprudence | — | — | — | — | — |
| medical_genetics | — | — | — | — | — |
| world_religions | — | — | — | — | — |
| moral_disputes | — | — | — | — | — |

⬜ **PENDING**

### Q3 Check: All categories rise?
⬜ **PENDING**

### Q4 Check: G > C + broad improvement?
⬜ **PENDING**

---

## MMLU Results — Full 57 Subjects

### Humanities (13 subjects)

| Subject | A | C | G | C-A | G-A |
|---------|---|---|---|-----|-----|
| formal_logic | — | — | — | — | — |
| high_school_european_history | — | — | — | — | — |
| high_school_us_history | — | — | — | — | — |
| high_school_world_history | — | — | — | — | — |
| international_law | — | — | — | — | — |
| jurisprudence | — | — | — | — | — |
| logical_fallacies | — | — | — | — | — |
| moral_disputes | — | — | — | — | — |
| moral_scenarios | — | — | — | — | — |
| philosophy | — | — | — | — | — |
| prehistory | — | — | — | — | — |
| professional_law | — | — | — | — | — |
| world_religions | — | — | — | — | — |

### Social Sciences (12 subjects)

| Subject | A | C | G | C-A | G-A |
|---------|---|---|---|-----|-----|
| econometrics | — | — | — | — | — |
| high_school_geography | — | — | — | — | — |
| high_school_gov_politics | — | — | — | — | — |
| high_school_macroeconomics | — | — | — | — | — |
| high_school_microeconomics | — | — | — | — | — |
| high_school_psychology | — | — | — | — | — |
| human_sexuality | — | — | — | — | — |
| professional_psychology | — | — | — | — | — |
| public_relations | — | — | — | — | — |
| security_studies | — | — | — | — | — |
| sociology | — | — | — | — | — |
| us_foreign_policy | — | — | — | — | — |

### Other (13 subjects)

| Subject | A | C | G | C-A | G-A |
|---------|---|---|---|-----|-----|
| business_ethics | — | — | — | — | — |
| clinical_knowledge | — | — | — | — | — |
| college_medicine | — | — | — | — | — |
| global_facts | — | — | — | — | — |
| human_aging | — | — | — | — | — |
| management | — | — | — | — | — |
| marketing | — | — | — | — | — |
| medical_genetics | — | — | — | — | — |
| miscellaneous | — | — | — | — | — |
| nutrition | — | — | — | — | — |
| professional_accounting | — | — | — | — | — |
| professional_medicine | — | — | — | — | — |
| virology | — | — | — | — | — |

### STEM (19 subjects)

| Subject | A | C | G | C-A | G-A |
|---------|---|---|---|-----|-----|
| abstract_algebra | — | — | — | — | — |
| anatomy | — | — | — | — | — |
| astronomy | — | — | — | — | — |
| college_biology | — | — | — | — | — |
| college_chemistry | — | — | — | — | — |
| college_computer_science | — | — | — | — | — |
| college_mathematics | — | — | — | — | — |
| college_physics | — | — | — | — | — |
| computer_security | — | — | — | — | — |
| conceptual_physics | — | — | — | — | — |
| electrical_engineering | — | — | — | — | — |
| elementary_mathematics | — | — | — | — | — |
| high_school_biology | — | — | — | — | — |
| high_school_chemistry | — | — | — | — | — |
| high_school_computer_science | — | — | — | — | — |
| high_school_mathematics | — | — | — | — | — |
| high_school_physics | — | — | — | — | — |
| high_school_statistics | — | — | — | — | — |
| machine_learning | — | — | — | — | — |

---

## Sign Test Summary

### G vs A
- Subjects improved: —
- Subjects declined: —
- Subjects unchanged: —
- p-value: —

### C vs A
- Subjects improved: —
- Subjects declined: —
- Subjects unchanged: —
- p-value: —

---

## Group 3 Verdict

| Question | Result | Detail |
|----------|--------|--------|
| Q1: C > A? | ⬜ PENDING | |
| Q2: Understanding/judgment rise? | ⬜ PENDING | |
| Q3: All categories rise? | ⬜ PENDING | |
| Q4: G > C + broad improvement? | ⬜ PENDING | |

**Group 3 Status: NOT STARTED**
