import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT/'data'/'classified_all.json', encoding='utf-8') as f:
    qs = json.load(f)

# aggregate per call
calls = {}
for q in qs:
    key = (q['ticker'], q['date'])
    c = calls.setdefault(key, {'ticker': q['ticker'], 'date': q['date'],
                               'return_5d': q.get('return_5d'),
                               'return_10d': q.get('return_10d'),
                               'labels': []})
    if q.get('predicted'):
        c['labels'].append(q['predicted'])

rows = []
for c in calls.values():
    n = len(c['labels'])
    if n < 5 or c['return_5d'] is None:
        continue
    counts = Counter(c['labels'])
    rows.append({
        'ticker': c['ticker'], 'date': c['date'], 'n': n,
        'alarm_share': counts['Alarmed'] / n,
        'negative_share': (counts['Alarmed'] + counts['Hedging']) / n,
        'confident_share': counts['Confident'] / n,
        'return_5d': c['return_5d'], 'return_10d': c['return_10d'],
    })

print(f"Calls analyzed: {len(rows)}")

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs) ** 0.5
    dy = sum((y-my)**2 for y in ys) ** 0.5
    return num/(dx*dy) if dx and dy else 0

# winsorize returns at 1st/99th percentile — a few extreme moves would dominate otherwise
def winsorize(vals, p=0.01):
    s = sorted(vals)
    lo, hi = s[int(len(s)*p)], s[int(len(s)*(1-p))]
    return [min(max(v, lo), hi) for v in vals]

for ret_key in ['return_5d', 'return_10d']:
    raw = [r[ret_key] for r in rows if r[ret_key] is not None]
    idx = [i for i, r in enumerate(rows) if r[ret_key] is not None]
    wins = winsorize(raw)
    print(f"\n--- {ret_key} ---")
    for share in ['alarm_share', 'negative_share', 'confident_share']:
        xs = [rows[i][share] for i in idx]
        print(f"  {share:18s} r = {pearson(xs, wins):+.3f}  (raw {pearson(xs, raw):+.3f})")

# quintile comparison — more interpretable than r
print("\n--- Mean 5d return by alarm-share quintile ---")
ranked = sorted([r for r in rows if r['return_5d'] is not None], key=lambda r: r['alarm_share'])
q = len(ranked) // 5
for i in range(5):
    bucket = ranked[i*q:(i+1)*q] if i < 4 else ranked[4*q:]
    mean_ret = sum(winsorize([b['return_5d'] for b in bucket]))/len(bucket)
    print(f"  Q{i+1} (alarm {bucket[0]['alarm_share']:.0%}–{bucket[-1]['alarm_share']:.0%}): "
          f"mean {mean_ret:+.2f}%  n={len(bucket)}")

with open(ROOT/'data'/'call_level_analysis.json', 'w') as f:
    json.dump(rows, f, indent=2)
print("\nSaved data/call_level_analysis.json")