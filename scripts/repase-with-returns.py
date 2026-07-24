import json
from datasets import load_dataset
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ds = load_dataset(
    "kunhanw/earning_call_transcript_dataset_with_volatility_analysis",
    cache_dir="C:/hf_cache"
)

# Build ticker+date → returns lookup
returns = {}
for row in ds['train']:
    meta = row['meta']
    vol = meta.get('volatility_analysis') or {}
    d5, d10 = vol.get('5_day') or {}, vol.get('10_day') or {}
    key = (meta.get('name',''), (meta.get('publishOn') or '')[:10])
    returns[key] = {
        'return_5d': d5.get('return_pct'),
        'return_10d': d10.get('return_pct'),
        'volatility_5d': d5.get('volatility_annual_pct'),
    }

with open(ROOT/'data'/'analyst_questions.json', encoding='utf-8') as f:
    questions = json.load(f)

matched = 0
for q in questions:
    r = returns.get((q['ticker'], q['date']))
    if r:
        q.update(r)
        matched += 1

print(f"Questions: {len(questions)}")
print(f"Matched to returns: {matched} ({matched/len(questions):.0%})")

with open(ROOT/'data'/'questions_with_returns.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2)
print("Saved data/questions_with_returns.json")