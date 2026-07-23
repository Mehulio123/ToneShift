import json
from collections import defaultdict

LABELS = ["Confident", "Cautious", "Hedging", "Alarmed"]

d = json.load(open("eval_results.json"))

def macro_f1(preds, tag):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p in preds:
        t, pr = p["true"], p["pred"]
        if t == pr:
            stats[t]["tp"] += 1
        else:
            stats[t]["fn"] += 1
            if pr in LABELS:
                stats[pr]["fp"] += 1

    f1s = []
    print(f"\n=== {tag} ===")
    for l in LABELS:
        s = stats[l]
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        f1s.append(f1)
        print(f"  {l:10s} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    m = sum(f1s) / len(f1s)
    print(f"  Macro-F1: {m:.3f}")
    return m

b = macro_f1(d["base_preds"], "BASE Mistral 7B")
t = macro_f1(d["tuned_preds"], "FINE-TUNED ToneShift")
print(f"\nMacro-F1 delta: {t - b:+.3f}")

# Degenerate baseline: what accuracy does always guessing the majority class get?
n = len(d["tuned_preds"])
maj = max(sum(1 for p in d["tuned_preds"] if p["true"] == l) for l in LABELS)
print(f"Always-majority-class baseline: {maj/n:.1%}")

# How often are errors adjacent on the intensity scale vs far apart?
order = {l: i for i, l in enumerate(LABELS)}
dists = [abs(order[p["true"]] - order[p["pred"]])
         for p in d["tuned_preds"] if p["pred"] in order and p["true"] != p["pred"]]
if dists:
    print(f"\nErrors 1 step away: {sum(1 for x in dists if x == 1)}/{len(dists)}")
    print(f"Errors 2+ steps away: {sum(1 for x in dists if x >= 2)}/{len(dists)}")