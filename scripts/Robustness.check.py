import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT/'data'/'classified_all.json', encoding='utf-8') as f:
    qs = json.load(f)

calls = {}
for q in qs:
    key = (q['ticker'], q['date'])
    c = calls.setdefault(key, {'ticker': q['ticker'], 'date': q['date'],
                               'return_5d': q.get('return_5d'),
                               'volatility_5d': q.get('volatility_5d'),
                               'labels': []})
    if q.get('predicted'):
        c['labels'].append(q['predicted'])

def pearson(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs)**0.5
    dy = sum((y-my)**2 for y in ys)**0.5
    return num/(dx*dy) if dx and dy else 0

def winsorize(vals, p=0.01):
    s = sorted(vals); lo, hi = s[int(len(s)*p)], s[int(len(s)*(1-p))]
    return [min(max(v, lo), hi) for v in vals]

for min_q in (5, 15):
    rows = []
    for c in calls.values():
        n = len(c['labels'])
        if n < min_q or c['return_5d'] is None:
            continue
        counts = Counter(c['labels'])
        rows.append({
            'alarm_share': counts['Alarmed']/n,
            'negative_share': (counts['Alarmed']+counts['Hedging'])/n,
            'abs_return': abs(c['return_5d']),
            'volatility': c.get('volatility_5d'),
        })

    print(f"\n{'='*50}\nCalls with {min_q}+ questions: n={len(rows)}")

    absret = winsorize([r['abs_return'] for r in rows])
    for share in ('alarm_share', 'negative_share'):
        xs = [r[share] for r in rows]
        print(f"  {share:16s} vs |return|   r = {pearson(xs, absret):+.3f}")

    vol_rows = [r for r in rows if r['volatility'] is not None]
    if vol_rows:
        vols = winsorize([r['volatility'] for r in vol_rows])
        for share in ('alarm_share', 'negative_share'):
            xs = [r[share] for r in vol_rows]
            print(f"  {share:16s} vs volatility r = {pearson(xs, vols):+.3f}  (n={len(vol_rows)})")