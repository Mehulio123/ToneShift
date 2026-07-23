#moved to a 3 scheme collapse
import json
from collections import defaultdict

d = json.load(open("eval_results.json"))

SCHEMES = {
    "3-class (Confident / Cautious+Hedging / Alarmed)": {
        "Confident": "Positive", "Cautious": "Neutral",
        "Hedging": "Neutral", "Alarmed": "Alarmed"},
    "2-class (Confident+Cautious / Hedging+Alarmed)": {
        "Confident": "Positive", "Cautious": "Positive",
        "Hedging": "Negative", "Alarmed": "Negative"},
}

def score(preds, mapping, tag):
    labels = sorted(set(mapping.values()))
    stats = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
    correct = 0
    for p in preds:
        if p["pred"] not in mapping: continue
        t, pr = mapping[p["true"]], mapping[p["pred"]]
        if t == pr: stats[t]["tp"] += 1; correct += 1
        else:
            stats[t]["fn"] += 1; stats[pr]["fp"] += 1
    n = sum(1 for p in preds if p["pred"] in mapping)
    f1s = []
    for l in labels:
        s = stats[l]
        pr_ = s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"] else 0
        rc = s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"] else 0
        f1 = 2*pr_*rc/(pr_+rc) if pr_+rc else 0
        f1s.append(f1)
        print(f"    {l:10s} P={pr_:.2f} R={rc:.2f} F1={f1:.2f}")
    print(f"    Accuracy {correct/n:.1%} | Macro-F1 {sum(f1s)/len(f1s):.3f}")

for name, mapping in SCHEMES.items():
    print(f"\n=== {name} ===")
    print("  BASE:");  score(d["base_preds"],  mapping, "base")
    print("  TUNED:"); score(d["tuned_preds"], mapping, "tuned")