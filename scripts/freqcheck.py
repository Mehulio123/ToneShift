import csv
from collections import Counter

with open('labelled-sample.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

labels = [row['label'].strip() for row in rows]
counts = Counter(labels)

print("Label distribution:")
for label, count in counts.most_common():
    print(f"{label}: {count}")

print(f"\nTotal labeled: {len([l for l in labels if l])}")
print(f"Unlabeled: {len([l for l in labels if not l])}")