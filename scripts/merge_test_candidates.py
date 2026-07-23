import csv
from collections import Counter, defaultdict

VALID = ('confident', 'cautious', 'hedging', 'alarmed')
FIELDS = ['company', 'ticker', 'exchange', 'date', 'analyst_name', 'question', 'label']
TARGET_PER_CLASS = 30

with open('training_data.csv', 'r', encoding='utf-8') as f:
    train_qs = set(r['question'].strip() for r in csv.DictReader(f))

with open('test_set_Labelled.csv', 'r', encoding='utf-8') as f:
    existing = list(csv.DictReader(f))
    existing_qs = set(r['question'].strip() for r in existing)

for r in existing:
    r['label'] = r['label'].strip().capitalize()

current_counts = Counter(r['label'] for r in existing)
deficits = {
    label: max(0, TARGET_PER_CLASS - current_counts.get(label, 0))
    for label in ('Confident', 'Cautious', 'Hedging', 'Alarmed')
}

with open('Archive/candidates_to_label_Labelled.csv', 'r', encoding='utf-8') as f:
    candidates = list(csv.DictReader(f))

kept = []
skipped_blank = 0
skipped_invalid_label = 0
skipped_in_training = 0
skipped_duplicate = 0
skipped_deficit_filled = defaultdict(int)

for r in candidates:
    label = r['label'].strip()
    text = r['question'].strip()

    if not label:
        skipped_blank += 1
        continue
    if label.lower() not in VALID:
        print(f"  Skipping row with unrecognized label {label!r}: {text[:60]}")
        skipped_invalid_label += 1
        continue

    label = label.capitalize()

    if text in train_qs:
        print(f"  WARNING: row matches a training_data.csv question, skipping: {text[:60]}")
        skipped_in_training += 1
        continue
    if text in existing_qs:
        skipped_duplicate += 1
        continue
    if deficits[label] <= 0:
        skipped_deficit_filled[label] += 1
        continue

    kept.append({
        'company': r['company'],
        'ticker': r['ticker'],
        'exchange': r['exchange'],
        'date': r['date'],
        'analyst_name': r['analyst_name'],
        'question': r['question'],
        'label': label,
    })
    existing_qs.add(text)
    deficits[label] -= 1

updated = existing + kept

with open('test_set_Labelled.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(updated)

print()
print(f"Added {len(kept)} new rows to test_set_Labelled.csv")
print(f"  skipped (left blank): {skipped_blank}")
print(f"  skipped (invalid label): {skipped_invalid_label}")
print(f"  skipped (matched training_data.csv): {skipped_in_training}")
print(f"  skipped (duplicate of existing test row): {skipped_duplicate}")
for label, n in skipped_deficit_filled.items():
    print(f"  skipped ({label} already at target of {TARGET_PER_CLASS}, extra labeled rows unused): {n}")
print()
print(f"test_set_Labelled.csv total rows: {len(updated)}")
print("New label distribution:")
final_counts = Counter(r['label'] for r in updated)
for label, count in final_counts.most_common():
    flag = "" if count >= TARGET_PER_CLASS else f"  <- still {TARGET_PER_CLASS - count} short of target"
    print(f"  {label}: {count}{flag}")
