import csv
from collections import Counter

with open('test_set_Labelled.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

with open('training_data.csv', 'r', encoding='utf-8') as f:
    train_qs = set(r['question'].strip() for r in csv.DictReader(f))

VALID = ('confident', 'cautious', 'hedging', 'alarmed')

clean = [
    r for r in rows
    if r['question'].strip() not in train_qs
    and r['label'].strip().lower() in VALID
]

for r in clean:
    r['label'] = r['label'].strip().capitalize()

with open('test_set.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['company','ticker','exchange','date','analyst_name','question','label'])
    writer.writeheader()
    writer.writerows(clean)

print(f"Before: {len(rows)}  →  After: {len(clean)}")
for label, count in Counter(r['label'] for r in clean).most_common():
    print(f"  {label}: {count}")