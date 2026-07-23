import csv

with open('test_set_Labelled.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys()

for row in rows:
    row['label'] = row['label'].strip().capitalize()

with open('test_set.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Labels normalized")