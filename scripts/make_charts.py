import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

with open(ROOT / "data" / "call_level_analysis.json", encoding="utf-8") as f:
    rows = [r for r in json.load(f) if r.get("return_5d") is not None]

def winsorize(vals, p=0.01):
    s = sorted(vals)
    lo, hi = s[int(len(s) * p)], s[int(len(s) * (1 - p))]
    return [min(max(v, lo), hi) for v in vals]

x = [r["alarm_share"] for r in rows]
y = winsorize([r["return_5d"] for r in rows])

# ---------- Chart 1: scatter with fit line ----------
n = len(x)
mx, my = sum(x)/n, sum(y)/n
sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
sxx = sum((a-mx)**2 for a in x)
slope = sxy/sxx if sxx else 0
intercept = my - slope*mx
r = sxy / ((sxx**0.5) * (sum((b-my)**2 for b in y)**0.5))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=12, alpha=0.35, color="#4a6fa5", edgecolors="none")
xs = [min(x), max(x)]
ax.plot(xs, [slope*v + intercept for v in xs], color="#c1444a", lw=2,
        label=f"fit: r = {r:+.3f}")
ax.axhline(0, color="#999", lw=0.8, ls="--")
ax.set_xlabel("Share of analyst questions classified Alarmed")
ax.set_ylabel("5-day forward return (%)")
ax.set_title(f"Analyst alarm vs. forward return — {n:,} earnings calls", pad=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "alarm_vs_return.png", dpi=150)
print("charts/alarm_vs_return.png")

# ---------- Chart 2: quintile bars ----------
ranked = sorted(rows, key=lambda r: r["alarm_share"])
q = len(ranked) // 5
means, labels = [], []
for i in range(5):
    bucket = ranked[i*q:(i+1)*q] if i < 4 else ranked[4*q:]
    means.append(sum(winsorize([b["return_5d"] for b in bucket])) / len(bucket))
    labels.append(f"Q{i+1}\n{bucket[0]['alarm_share']:.0%}–{bucket[-1]['alarm_share']:.0%}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, means, color=["#c1444a" if m < 0 else "#4a8f5f" for m in means], width=0.6)
ax.axhline(0, color="#333", lw=1)
ax.set_ylabel("Mean 5-day return (%)")
ax.set_xlabel("Alarm-share quintile (low → high)")
ax.set_title("No monotonic relationship between alarm and return", pad=12)
for i, m in enumerate(means):
    ax.text(i, m + (0.15 if m >= 0 else -0.35), f"{m:+.2f}%", ha="center", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "quintile_returns.png", dpi=150)
print("charts/quintile_returns.png")

# ---------- Chart 3: per-class F1, base vs tuned ----------
try:
    with open(ROOT / "eval_results.json", encoding="utf-8") as f:
        ev = json.load(f)
except FileNotFoundError:
    print("eval_results.json not found — skipping chart 3")
    raise SystemExit

LABELS = ["Confident", "Cautious", "Hedging", "Alarmed"]

def f1s(preds):
    out = []
    for lab in LABELS:
        tp = sum(1 for p in preds if p["true"] == lab and p["pred"] == lab)
        fp = sum(1 for p in preds if p["true"] != lab and p["pred"] == lab)
        fn = sum(1 for p in preds if p["true"] == lab and p["pred"] != lab)
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        out.append(2*prec*rec/(prec+rec) if prec+rec else 0)
    return out

base, tuned = f1s(ev["base_preds"]), f1s(ev["tuned_preds"])
pos = range(len(LABELS))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([p - 0.2 for p in pos], base, width=0.4, label="Base Mistral 7B", color="#b8b8b8")
ax.bar([p + 0.2 for p in pos], tuned, width=0.4, label="Fine-tuned", color="#4a6fa5")
ax.set_xticks(list(pos)); ax.set_xticklabels(LABELS)
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1 — base vs. fine-tuned", pad=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for i, (b, t) in enumerate(zip(base, tuned)):
    ax.text(i - 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.text(i + 0.2, t + 0.01, f"{t:.2f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "per_class_f1.png", dpi=150)
print("charts/per_class_f1.png")