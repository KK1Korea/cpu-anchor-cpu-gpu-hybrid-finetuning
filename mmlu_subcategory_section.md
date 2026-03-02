## What changed inside MMLU?

Overall MMLU shifted +0.09% — within noise. But the subcategory breakdown tells a different story.

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

### One more question

If the selective improvement in understanding and judgment is real — does it extend to long-context coherence? A model that genuinely "understands" should maintain context over thousands of tokens better than one relying on pattern matching. Needle-in-a-haystack and multi-turn consistency tests could reveal differences that MMLU's short-form questions cannot capture.

If CPU anchoring genuinely separates 'what to learn precisely' from 'what to interpolate' — what happens when the training data itself is structured that way? Anchor-type data (clear definitions, boundaries) for the CPU phase, interpolation-type data (contextual reasoning, nuance) for the GPU phase. Does deliberate data separation amplify the effect beyond hardware switching alone?
