import csv
from collections import Counter

with open('test_set_Labelled.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

counts = Counter(row['label'].strip() for row in rows)
for label, count in counts.most_common():
    print(f"{label}: {count}")