import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT/'data'/'questions_with_returns.json', encoding='utf-8') as f:
    qs = json.load(f)

calls = {}
for q in qs:
    calls.setdefault((q['ticker'], q['date']), []).append(q)

keep = [q for k, v in calls.items() if len(v) >= 5 for q in v]
print(f"Questions to classify: {len(keep)}")
print(f"At ~1.5s each: {len(keep)*1.5/3600:.1f} hours")
print(f"At $0.40/hr: ${len(keep)*1.5/3600*0.40:.2f}")

with open(ROOT/'data'/'to_classify.json', 'w', encoding='utf-8') as f:
    json.dump(keep, f)
print("Saved data/to_classify.json")