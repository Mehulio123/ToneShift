import json
from collections import Counter

with open('data/analyst_questions.json', encoding='utf-8') as f:
    qs = json.load(f)

months = Counter(q['date'][:7] for q in qs if q.get('date'))
print(f"Total questions: {len(qs)}")
print(f"Date range: {min(months)} to {max(months)}\n")
for m in sorted(months):
    print(f"  {m}: {months[m]}")

# how many tickers have 3+ distinct months?
by_ticker = {}
for q in qs:
    by_ticker.setdefault(q['ticker'], set()).add(q['date'][:7])
multi = {t: len(m) for t, m in by_ticker.items() if len(m) >= 3}
print(f"\nTickers with 3+ distinct months: {len(multi)}")
for t, n in sorted(multi.items(), key=lambda x: -x[1])[:15]:
    print(f"  {t}: {n} months")