# ToneShift — Evaluation Results

Fine-tuned Mistral 7B (QLoRA) for classifying analyst sentiment tone in earnings call questions.

## Setup

- **Base model:** Mistral-7B-Instruct-v0.3, 4-bit quantized (NF4)
- **Method:** QLoRA — r=16, alpha=32, target modules q_proj/v_proj, 3 epochs, lr 2e-4, bf16
- **Training set:** 574 hand-labeled analyst questions
- **Test set:** 121 held-out questions, balanced across classes (30/31/30/30), verified zero overlap with training data
- **Comparison:** identical prompt, identical test set, base model vs. fine-tuned adapter

## Headline result

| Scheme | Base Macro-F1 | Fine-tuned Macro-F1 | Relative lift |
|---|---|---|---|
| 4-class (Confident/Cautious/Hedging/Alarmed) | 0.170 | 0.358 | +111% |
| 3-class (Confident/Neutral/Alarmed) | 0.266 | 0.472 | +77% |
| 2-class (Positive/Negative) | 0.405 | 0.605 | +49% |

Macro-F1 is reported rather than accuracy because accuracy is misleading on this task — see below.

## Why accuracy is the wrong metric here

The base model collapses to majority-class prediction. On the 4-class test set it labeled nearly everything "Cautious":

| True class | Base model predictions |
|---|---|
| Confident (30) | Cautious 26, Confident 2, Hedging 1, unparseable 1 |
| Cautious (31) | Cautious 28, Hedging 2, Confident 1 |
| Hedging (30) | Cautious 27, Hedging 3 |
| Alarmed (30) | Cautious 27, Hedging 2, Confident 1 |

This scores 27.3% accuracy — roughly the base rate of the largest class. The model is not classifying; it is guessing one label. Macro-F1 (0.170) exposes this where accuracy does not.

In the 3-class scheme the effect is stark: the base model's accuracy (51.7%) is *higher* than the fine-tuned model's (50.0%), while its macro-F1 is far lower (0.266 vs 0.472). The fine-tuned model loses raw accuracy precisely because it stops defaulting to the majority class.

## Per-class results — 3-class scheme

| Class | Base P / R / F1 | Fine-tuned P / R / F1 |
|---|---|---|
| Confident | 0.50 / 0.07 / 0.12 | 0.39 / 0.45 / 0.42 |
| Neutral | 0.52 / 0.98 / 0.68 | 0.52 / 0.61 / 0.56 |
| Alarmed | 0.00 / 0.00 / 0.00 | 0.62 / 0.33 / 0.43 |

**Alarmed is the class this project exists to detect.** The base model identified zero of 30 alarmed questions correctly. The fine-tuned model reaches 0.62 precision — its highest of any class — meaning when it flags alarm, it is right roughly 62% of the time. Recall of 0.33 is the honest limitation: it is a high-confidence detector with incomplete coverage.

## Errors follow the intensity gradient

Of 77 misclassifications in the 4-class scheme:

- **49 (64%)** were one step away on the Confident → Cautious → Hedging → Alarmed scale
- **28 (36%)** were two or more steps away

The model rarely confuses Confident with Alarmed. It has learned the underlying intensity gradient but places soft boundaries between adjacent categories — consistent with the fact that the Cautious/Hedging distinction is genuinely fuzzy in the source data.

## Design decisions and limitations

**Granularity.** Fine-tuning was performed on 4 classes; the deployed system collapses Cautious and Hedging into a single Neutral class at inference. This boundary was the primary source of error and was not consistently separable. Alarmed is preserved as its own class because it carries the signal the project targets. Training directly on 3 classes would likely exceed 0.472 and is listed as future work.

**Test set balance.** The test set is balanced across classes rather than matching the real-world distribution (where neutral-toned questions dominate). This measures whether the model can distinguish the classes, not what the base rates are, and prevents a majority-class guesser from scoring well.

**Label consistency.** Training and test sets were labeled in separate sessions. Their class distributions differ, which suggests some labeling drift between sessions. Some measured errors may reflect inconsistency in the ground truth rather than model failure. A self-agreement audit would bound this and has not yet been run.

**Parser noise.** The transcript parser occasionally admits management responses and closing remarks rather than analyst questions. These were filtered manually from the test set but remain present in the full corpus at an unmeasured rate.

**Scale.** 574 training examples is small. 121 test examples means a single prediction moves a per-class metric by roughly 3 points.

## Reproducing

```bash
python scripts/clean_test_set.py     # verify no train/test contamination
python scripts/train.py              # RunPod, ~5 min on A40
python scripts/eval.py               # RunPod, ~10 min — writes eval_results.json
python scripts/macro_f1.py           # local
python scripts/collapse_eval.py      # local — granularity comparison
```

## Cost

Total GPU spend: approximately $15.8 on RunPod (A40, $0.40/hr). Training run: ~5 minutes. Full evaluation: ~10 minutes.

---
*Last updated: 23rd July 2026*