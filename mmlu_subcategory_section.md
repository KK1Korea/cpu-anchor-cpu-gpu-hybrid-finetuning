## What changed inside MMLU?

Overall MMLU shifted +0.09% (A→C) — within noise. But the subcategory breakdown tells a different story.

**Subjects that improved (≥ +1.0%):**

| Subject | Exp A | Exp C | Change | Category |
|:--|:--|:--|:--|:--|
| high_school_physics | 0.5695 | 0.5894 | **+1.99%** | STEM |
| moral_scenarios | 0.6000 | 0.6150 | **+1.50%** | Humanities |
| world_religions | 0.8713 | 0.8830 | **+1.17%** | Humanities |
| prehistory | 0.8500 | 0.8600 | **+1.00%** | Humanities |
| human_aging | 0.7450 | 0.7550 | **+1.00%** | Other |
| medical_genetics | 0.8300 | 0.8400 | **+1.00%** | Other |
| professional_medicine | 0.7650 | 0.7750 | **+1.00%** | Other |
| professional_psychology | 0.7700 | 0.7800 | **+1.00%** | Social Sci |

**Subjects that declined (≤ -1.0%):**

| Subject | Exp A | Exp C | Change | Category |
|:--|:--|:--|:--|:--|
| college_mathematics | 0.4500 | 0.4300 | **-2.00%** | STEM |
| college_physics | 0.5098 | 0.4902 | **-1.96%** | STEM |
| high_school_computer_science | 0.8900 | 0.8800 | **-1.00%** | STEM |
| abstract_algebra | 0.5300 | 0.5200 | **-1.00%** | STEM |
| global_facts | 0.4600 | 0.4500 | **-1.00%** | Other |
| professional_law | 0.5650 | 0.5550 | **-1.00%** | Humanities |

**Category averages:**

| Category | Change |
|:--|:--|
| Humanities | **+0.34%** |
| Other | +0.12% |
| Social Sciences | +0.06% |
| STEM | **-0.26%** |

29 out of 57 subjects showed zero change.

### What does this mean?

I don't know. But the pattern is hard to ignore:

- **Improved:** moral judgment, religious understanding, medical reasoning, psychological assessment, physics intuition — subjects requiring contextual understanding.
- **Declined:** college math, abstract algebra, college physics (calculation-heavy), computer science — subjects solvable by pattern matching and formula application.

The CPU anchor didn't uniformly improve everything. It **redistributed** — away from calculation, toward comprehension.

This is a single seed, 500 steps, 7B model. The pattern needs multi-seed verification before drawing conclusions. But if it holds, the implication is that CPU anchoring doesn't just reduce train loss — it changes **what the model is good at.**

Full per-subject scores for all 57 MMLU subjects are available in the repository.

---

## Update: Experiments G and F answered the redistribution question (2026-03-02)

Experiment G changed **one variable** from Experiment C: Phase2 precision bf16 → fp32. Same CPU anchor, same data, same steps.

Experiment F changed **one variable** from Experiment C: Phase1 device CPU → GPU. Same fp32 precision, same Phase2 bf16, same data, same steps.

**The result: redistribution was NOT the anchor. It was the interaction of CPU determinism + bf16.**

### A vs C vs F vs G Category Comparison

| Category | A (GPU bf16) | C (Anchor+bf16) | F (GPU fp32+bf16) | G (Anchor+fp32) | A→C | A→F | A→G |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Humanities | 78.04% | 78.35% | 78.39% | **78.53%** | +0.31 | +0.35 | **+0.49** |
| Social Sciences | 84.56% | 84.65% | 84.56% | **85.09%** | +0.09 | 0.00 | **+0.53** |
| Other | 74.70% | 74.84% | 74.79% | **74.98%** | +0.14 | +0.09 | **+0.28** |
| STEM | 69.79% | 69.65% ↓ | 69.97% ↑ | **70.16%** ↑ | -0.14 | **+0.18** | **+0.37** |
| **Overall** | **76.25%** | **76.34%** | **76.41%** | **76.66%** | +0.09 | +0.16 | **+0.41** |

**Ordering: A < C < F < G.** F outperformed C despite lacking CPU determinism.

Critical STEM finding: C declined (-0.14%) while F improved (+0.18%). Both used bf16 in Phase2 — the only difference was CPU (deterministic) vs GPU (non-deterministic) in Phase1.

### The calculation subjects recovered — but differently

| Subject | A | C (bf16) | F (GPU fp32) | G (CPU fp32) | A→C | A→F | A→G |
|:--|:--|:--|:--|:--|:--|:--|:--|
| college_mathematics | 45.00% | 43.00% ↓ | 43.00% ↓ | **47.00%** ↑ | -2.00 | -2.00 | **+2.00** |
| abstract_algebra | 53.00% | 52.00% ↓ | **53.00%** ✅ | **53.00%** | -1.00 | 0.00 | 0.00 |
| college_physics | 50.98% | 49.02% ↓ | **50.98%** ✅ | 50.00% | -1.96 | 0.00 | -0.98 |
| high_school_computer_science | 89.00% | 88.00% ↓ | **89.00%** ✅ | **90.00%** ↑ | -1.00 | 0.00 | **+1.00** |
| high_school_chemistry | 66.50% | 66.00% ↓ | **66.50%** ✅ | **68.00%** ↑ | -0.50 | 0.00 | **+1.50** |

C declined in 5/5 subjects. **F recovered 4/5 to baseline** (only college_math remained down). G recovered or surpassed all 5.

The pattern: CPU determinism + bf16 = STEM decline. GPU non-determinism + bf16 = STEM preserved. CPU determinism + fp32 = STEM recovered and surpassed.

### The understanding subjects held or grew further

| Subject | A | C (bf16) | F (GPU fp32) | G (CPU fp32) | A→C | A→F | A→G |
|:--|:--|:--|:--|:--|:--|:--|:--|
| moral_scenarios | 60.00% | 61.50% ↑ | **63.00%** ↑↑ | **64.50%** ↑↑↑ | +1.50 | **+3.00** | **+4.50** |
| high_school_physics | 56.95% | 58.94% ↑ | 58.28% ↑ | **59.60%** ↑ | +1.99 | +1.33 | **+2.65** |
| professional_psychology | 77.00% | 78.00% ↑ | 77.50% | **79.00%** ↑ | +1.00 | +0.50 | **+2.00** |
| us_foreign_policy | 88.00% | 88.00% | 89.00% ↑ | **90.00%** ↑ | 0.00 | +1.00 | **+2.00** |
| human_sexuality | 79.39% | 78.63% ↓ | **80.15%** ↑ | **80.92%** ↑ | -0.76 | +0.76 | **+1.53** |
| medical_genetics | 83.00% | 84.00% ↑ | **84.00%** | **84.00%** | +1.00 | +1.00 | +1.00 |
| jurisprudence | 80.56% | 81.48% ↑ | **81.48%** | **81.48%** | +0.92 | +0.92 | +0.92 |

Anchor-driven understanding gains **persisted across all four experiments**. moral_scenarios formed a perfect staircase: A (60.0%) → C (61.5%) → F (63.0%) → G (64.5%) — each experiment adding exactly +1.5%.

### Full 57-subject A vs G comparison

**Improved (27 subjects):**

| Subject | A | G | Δ | Category |
|:--|:--|:--|:--|:--|
| moral_scenarios | 60.00 | 64.50 | **+4.50** | Humanities |
| high_school_physics | 56.95 | 59.60 | **+2.65** | STEM |
| college_mathematics | 45.00 | 47.00 | **+2.00** | STEM |
| professional_psychology | 77.00 | 79.00 | **+2.00** | Social Sci |
| us_foreign_policy | 88.00 | 90.00 | **+2.00** | Social Sci |
| human_sexuality | 79.39 | 80.92 | **+1.53** | Social Sci |
| high_school_mathematics | 60.50 | 62.00 | +1.50 | STEM |
| high_school_chemistry | 66.50 | 68.00 | +1.50 | STEM |
| human_aging | 74.50 | 76.00 | +1.50 | Other |
| business_ethics | 78.00 | 79.00 | +1.00 | Other |
| global_facts | 46.00 | 47.00 | +1.00 | Other |
| medical_genetics | 83.00 | 84.00 | +1.00 | Other |
| professional_accounting | 55.00 | 56.00 | +1.00 | Other |
| professional_medicine | 76.50 | 77.50 | +1.00 | Other |
| security_studies | 79.50 | 80.50 | +1.00 | Social Sci |
| college_computer_science | 66.00 | 67.00 | +1.00 | STEM |
| high_school_computer_science | 89.00 | 90.00 | +1.00 | STEM |
| jurisprudence | 80.56 | 81.48 | +0.92 | Humanities |
| international_law | 84.30 | 85.12 | +0.82 | Humanities |
| logical_fallacies | 78.53 | 79.14 | +0.61 | Humanities |
| world_religions | 87.13 | 87.72 | +0.59 | Humanities |
| high_school_us_history | 88.50 | 89.00 | +0.50 | Humanities |
| prehistory | 85.00 | 85.50 | +0.50 | Humanities |
| high_school_macroeconomics | 82.50 | 83.00 | +0.50 | Social Sci |
| sociology | 88.50 | 89.00 | +0.50 | Social Sci |
| high_school_biology | 87.00 | 87.50 | +0.50 | STEM |
| high_school_statistics | 71.50 | 72.00 | +0.50 | STEM |

**Unchanged (16 subjects):**

formal_logic, high_school_european_history, clinical_knowledge, college_medicine, marketing, miscellaneous, high_school_geography, high_school_government_and_politics, high_school_microeconomics, high_school_psychology, public_relations, abstract_algebra, astronomy, college_biology, computer_security, electrical_engineering

**Declined (14 subjects):**

| Subject | A | G | Δ | Category |
|:--|:--|:--|:--|:--|
| nutrition | 80.50 | 79.50 | -1.00 | Other |
| college_chemistry | 54.00 | 53.00 | -1.00 | STEM |
| college_physics | 50.98 | 50.00 | -0.98 | STEM |
| management | 88.35 | 87.38 | -0.97 | Other |
| machine_learning | 60.71 | 59.82 | -0.89 | STEM |
| econometrics | 68.42 | 67.54 | -0.88 | Social Sci |
| anatomy | 71.11 | 70.37 | -0.74 | STEM |
| virology | 55.42 | 54.82 | -0.60 | Other |
| high_school_world_history | 89.50 | 89.00 | -0.50 | Humanities |
| moral_disputes | 77.00 | 76.50 | -0.50 | Humanities |
| philosophy | 81.50 | 81.00 | -0.50 | Humanities |
| professional_law | 56.50 | 56.00 | -0.50 | Humanities |
| conceptual_physics | 74.50 | 74.00 | -0.50 | STEM |
| elementary_mathematics | 71.00 | 70.50 | -0.50 | STEM |

**Summary: 27 improved, 16 unchanged, 14 declined. Ratio ~2:1 in favor of improvement.**
**Average improvement: +1.16%. Average decline: -0.72%. Net direction: positive across all categories.**

### The three-factor model

The four-way verification (A, C, F, G) reveals three independent factors:

**Factor 1: fp32 Precision Anchor (dominant)**
- Running first 20% of steps in fp32 improves understanding, judgment, and reasoning subjects
- Present in C, F, and G (all have fp32 Phase1)
- Accounts for 96.6% of train loss improvement (A→F)
- Strongest signal: moral_scenarios, high_school_physics, medical_genetics

**Factor 2: CPU Determinism (secondary, double-edged)**
- Deepens the anchor (enables sub-1.0 train loss basins)
- BUT suppresses STEM: C STEM 69.65% < A STEM 69.79%, while F STEM 69.97% > A
- Present in C and G (both have CPU Phase1), absent in F
- Accounts for 3.4% of train loss improvement (F→C)
- Redistributes capability: understanding↑, calculation↓

**Factor 3: Phase2 Precision (bf16 vs fp32)**
- Invisible in train loss (C ≈ G, Δ 0.12%)
- Visible in MMLU: G outperformed C in all categories (+0.32%)
- Recovers the STEM penalty from Factor 2
- Strongest signal: college_mathematics, high_school_chemistry, moral_scenarios

Experiment A = none → baseline
Experiment F = Factor 1 only → +0.16% MMLU, STEM preserved
Experiment C = Factor 1 + Factor 2 → +0.09% MMLU, STEM declined (partial cancellation)
Experiment G = Factor 1 + Factor 2 + Factor 3 → **+0.41%** MMLU, all categories improved

**The "redistribution" observed in C was the interaction of Factors 1 and 2.** The fp32 anchor improved understanding. CPU determinism added depth but suppressed STEM. When GPU non-deterministic fp32 is used instead (Experiment F), understanding improves *and* STEM is preserved — no redistribution at all.

### The deepest finding

Train loss: C = 0.9177, F = 0.9268, G = 0.9188. Spread: 0.99%. Small.

MMLU: C = 76.34%, F = 76.41%, G = 76.66%. **F beat C with worse train loss.**

This breaks the assumption that lower train loss = better model. C had the lowest train loss of all four experiments but ranked third in MMLU. F had the highest train loss among the anchored experiments but ranked second. **Train loss and MMLU are measuring different things.**

What matters more than how low the loss goes is *how the model arrives there*. CPU determinism pushes loss deeper (sub-1.0) but constrains the parameter space in ways that suppress STEM. GPU non-determinism finds a shallower basin but preserves a more balanced capability profile.

This means the anchor doesn't just change how efficiently the model learns. It changes **what the model becomes.** Phase1 device determines **how the anchor constrains exploration.** Phase2 precision determines **how clearly that identity is expressed.**

### One more question

If the selective improvement in understanding and judgment is real — does it extend to long-context coherence? A model that genuinely "understands" should maintain context over thousands of tokens better than one relying on pattern matching. Needle-in-a-haystack and multi-turn consistency tests could reveal differences that MMLU's short-form questions cannot capture.

If CPU anchoring genuinely separates 'what to learn precisely' from 'what to interpolate' — what happens when the training data itself is structured that way? Anchor-type data (clear definitions, boundaries) for the CPU phase, interpolation-type data (contextual reasoning, nuance) for the GPU phase. Does deliberate data separation amplify the effect beyond hardware switching alone?

### New question from Experiment G

If 4-bit QLoRA masks train loss differences but NOT MMLU differences — what happens at 8-bit or 16-bit full precision? Does the MMLU gap between bf16 and fp32 widen when the quantization bottleneck is removed? A 3B model with 16-bit full loading could answer this within the same 24GB VRAM budget, while simultaneously testing whether a weaker base model with more MMLU headroom shows clearer differentiation.

---

## Update: Experiment F full subject analysis (2026-03-02)

Experiment F (GPU fp32 100 → GPU bf16 400) isolates fp32 precision from CPU determinism. This is the precision control for Experiment C, suggested by Damione (HuggingFace).

### F's unique subject-level behaviors

F achieved the **highest score of all four experiments** in 6 subjects:

| Subject | A | C | F | G | Category |
|:--|:--|:--|:--|:--|:--|
| world_religions | 87.13% | 88.30% | **89.47%** | 87.72% | Humanities |
| college_chemistry | 54.00% | 54.00% | **56.00%** | 53.00% | STEM |
| elementary_mathematics | 71.00% | 71.00% | **72.00%** | 70.50% | STEM |
| high_school_geography | 90.40% | 90.91% | **91.41%** | 90.40% | Social Sci |
| high_school_psychology | 91.00% | 91.00% | **91.50%** | 91.00% | Social Sci |
| virology | 55.42% | 55.42% | **56.02%** | 54.82% | Other |

These are not subjects that CPU-anchored C or G excelled in. GPU non-deterministic fp32 exploration found optimization paths that deterministic CPU never reached.

### The moral_scenarios staircase

| Experiment | Score | Δ from previous |
|:--|:--|:--|
| A (GPU bf16) | 60.00% | — |
| C (CPU fp32 → bf16) | 61.50% | +1.50 |
| F (GPU fp32 → bf16) | 63.00% | +1.50 |
| G (CPU fp32 → fp32) | 64.50% | +1.50 |

Perfect +1.50% increments. The ordering A < C < F < G suggests two additive effects:
- fp32 precision in Phase1 → +3.00% (A→F)
- Phase2 fp32 → additional +1.50% (F→G, roughly)
- CPU determinism → mixed effect (C between A and F)

### Why did F beat C on MMLU despite worse train loss?

| Metric | C | F |
|:--|:--|:--|
| Train loss | **0.9177** | 0.9268 |
| MMLU overall | 76.34% | **76.41%** |
| STEM | 69.65% | **69.97%** |
| Sub-1.0 achieved? | Yes (0.9966) | No (1.007) |

C went deeper (lower loss, broke sub-1.0) but narrower (STEM suppressed). F stayed shallower but wider (STEM preserved, unique subject peaks). 

Possible explanation: CPU determinism produces a single, deep, fixed optimization trajectory. Every run follows the exact same path. GPU non-determinism produces a broader exploration landscape in Phase1 — the 100 fp32 steps sample a wider region of parameter space before bf16 takes over. This wider sampling may explain why F found peaks in subjects (college_chemistry, elementary_mathematics) that the deterministic path missed.

### Revised redistribution narrative

Original (A vs C): *"The anchor redistributed — away from calculation, toward comprehension."*

Updated (A vs C vs F vs G):
- **fp32 precision** (Factor 1) → improves understanding AND preserves calculation. No redistribution.
- **CPU determinism** (Factor 2) → improves understanding BUT suppresses calculation. Redistribution.
- **Phase2 bf16** (Factor 3, penalty) → suppresses calculation further.
- **C = Factor 1 + Factor 2 + Factor 3** → appeared as redistribution, but was Factor 2+3 overlap.
- **F = Factor 1 only** → improvement without tradeoff. The "free lunch."
- **G = Factor 1 + Factor 2 - Factor 3** → Factor 3 removed by fp32, Factor 2's STEM penalty recovered.

The clean story: **Precision-Staged Training (F method) is pure gain. CPU anchoring (C method) adds depth at the cost of balance. Full fp32 (G method) restores balance at the cost of speed.**
