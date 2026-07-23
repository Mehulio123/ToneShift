import csv

with open('test_set.csv', 'r', encoding='utf-8') as f:
    test = list(csv.DictReader(f))
with open('training_data.csv', 'r', encoding='utf-8') as f:
    train_qs = set(r['question'].strip() for r in csv.DictReader(f))

overlap = [r for r in test if r['question'].strip() in train_qs]
print(f"Test rows: {len(test)}")
print(f"Also in training: {len(overlap)}")
for r in overlap[:5]:
    print(" -", r['ticker'], r['question'][:70])