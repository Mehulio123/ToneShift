# ToneShift

An empirical test of whether the tone of analyst questions on earnings calls predicts
short-term stock movement — using a fine-tuned LLM as the measurement instrument.

**Finding: it doesn't.** Across 1,527 earnings calls, the share of alarmed-tone analyst
questions showed no meaningful correlation with 5-day or 10-day forward returns
(r between -0.03 and +0.08), and the null held across four alternative specifications.

---

## The hypothesis

Analysts on earnings calls have money on the line and ask unscripted questions.
Management remarks are PR-written; the Q&A is not. The premise was that analyst
language shifts — from confident to hedging to alarmed — before price action does,
and that detecting the shift would surface a leading indicator.

Testing that required a classifier that could read tone in financial language.
No labeled dataset existed, so I built one and fine-tuned a model on it.

## What was built

**A tone classifier.** Mistral 7B fine-tuned with QLoRA on 574 hand-labeled analyst
questions, classifying tone as Confident, Cautious, Hedging, or Alarmed with a
one-sentence justification. Macro-F1 of 0.472 on a held-out balanced test set versus
0.266 for the base model — a 77% improvement. On the Alarmed class specifically, the
base model scored 0.00 precision (0 of 30 correct); the fine-tuned model reaches 0.62.

**A dataset.** 22,885 analyst questions parsed from 1,831 earnings call transcripts
across NYSE and NASDAQ, with forward returns joined from the source dataset's
volatility metadata. 574 questions hand-labeled for training, 121 for a held-out test
set verified to have zero overlap with training data.

**A cross-sectional analysis.** All 22,404 questions from calls with 5+ questions were
classified, aggregated to per-call tone shares, and correlated against forward returns.

## What happened to the hypothesis

The original design tracked sentiment drift for a single company across quarters.
That turned out to be impossible: the dataset spans 2024-07 to 2024-12, and exactly
one ticker appears in three or more distinct months. There is no time series to plot.

The hypothesis was reformulated as a cross-sectional test — across many companies in
the same window, does tone correlate with the subsequent move? — and tested. It does not.

| Predictor | 5-day return | 10-day return | \|return\| | Volatility |
|---|---|---|---|---|
| Alarm share | -0.012 | -0.032 | -0.064 | -0.078 |
| Negative share | -0.025 | -0.031 | -0.042 | -0.037 |
| Confident share | +0.063 | +0.075 | — | — |

Re-tested across alternative outcome variables and sample thresholds (5+ vs 15+
questions per call); all correlations stayed within ±0.08.

This is consistent with weak-form market efficiency. Earnings transcripts are public
and widely read — a simple tone signal being predictive would be the surprising result.

Full methodology, per-class metrics, confusion matrices, and limitations:
[RESULTS.md](RESULTS.md)

## Model performance

| Scheme | Base Macro-F1 | Fine-tuned | Lift |
|---|---|---|---|
| 4-class | 0.170 | 0.358 | +111% |
| 3-class | 0.266 | 0.472 | +77% |
| 2-class | 0.405 | 0.605 | +49% |

Macro-F1 rather than accuracy, because the base model collapses to majority-class
prediction — it labeled nearly every question "Cautious," scoring 27.3% accuracy on a
test set that is 26% Cautious. Accuracy rewards that; macro-F1 exposes it.

The model was trained on 4 classes and is deployed at 3-class granularity. The
Cautious/Hedging boundary was the dominant source of error and was not consistently
separable in the source data; Alarmed is preserved as its own class because it carries
the signal the project targets. 64% of remaining errors are between adjacent classes on
the intensity scale, indicating the model learned the underlying gradient but places
soft boundaries.

## Pipeline

```
HuggingFace earnings-call dataset (1,831 transcripts)
  → parse analyst Q&A, excluding management and operator turns   → 22,885 questions
  → hand-label 574 for training, 121 held out for testing
  → QLoRA fine-tune Mistral 7B (RunPod A40)                      → adapter, ~5 min
  → evaluate vs base model on held-out set                       → macro-F1, confusion
  → batch-classify 22,404 questions                              → ~1 hr, 16-way batched
  → aggregate to per-call tone shares, join forward returns
  → correlate, winsorize, robustness-check                       → null result
```
## Stack

**Model** — Mistral-7B-Instruct-v0.3, 4-bit NF4 quantization, QLoRA (r=16, α=32,
q_proj/v_proj), 3 epochs, lr 2e-4, bf16. HuggingFace Transformers, PEFT, TRL.

**Training** — RunPod A40. Total GPU spend across all runs: $18.80.

**Analysis** — Python, no ML dependencies. Pearson correlation and winsorization
implemented directly.

**Frontend** *(in progress)* — Next.js. Three views: a corpus explorer for browsing
calls and per-question classifications; a visualization of the null result (alarm-share
vs return scatter, quintile comparison); and a live classifier endpoint backed by a
HuggingFace Space.

## Running it

```bash
pip install -r requirements.txt

python scripts/parse_transcripts.py       # transcripts → analyst questions
python scripts/join_returns.py            # attach forward returns
python scripts/clean_test_set.py          # verify no train/test contamination
python scripts/train.py                   # RunPod, ~5 min on A40
python scripts/eval.py                    # RunPod, ~10 min → eval_results.json
python scripts/macro_f1.py                # local — per-class metrics
python scripts/collapse_eval.py           # local — granularity comparison
python scripts/classify_all.py            # RunPod, ~1 hr — batch inference
python scripts/analyze_correlation.py     # local — the test
python scripts/robustness_check.py        # local — alternative specifications
```

Training and inference scripts require a GPU. Everything else runs on CPU.

## What I'd do differently

**Check the dataset's time coverage before designing around it.** The longitudinal
drift design was built on an assumption I never verified. Counting distinct months per
ticker would have taken thirty seconds and saved a wrong turn.

**Define label criteria in writing before labeling.** Training and test sets were
labeled in separate sessions with different implicit standards, and their class
distributions diverge as a result. Some measured error is probably my own inconsistency
rather than model failure. Explicit written criteria — plus a self-agreement audit
before training — would bound this.

**Train on the granularity I intended to deploy.** Training on 4 classes and collapsing
to 3 at inference wastes capacity on a boundary that gets discarded. Retraining directly
on 3 classes would likely beat 0.472.

**Verify train/test separation from the start.** 22 contaminated rows made it into an
early test set through a file-naming mistake. Caught before it affected any reported
result, but a contamination check should have been part of the pipeline, not a reaction.

## Limitations

- Single earnings season (2024 H2) — not a full market cycle
- 574 training examples is small for a 4-class task
- 8% corpus-wide alarm rate means 40%+ of calls contain zero alarmed questions,
  limiting the resolution of alarm-share as a variable
- The transcript parser occasionally admits management responses and closing remarks;
  filtered manually from the test set, unmeasured in the full corpus
- Alarmed recall is 0.33 — a high-precision, low-coverage detector
- The null covers short horizons only. It does not test quarterly horizons, within-company
  drift, or tone relative to an analyst's own baseline. The dataset cannot support those.