import json
import random
import csv

with open('data/analyst_questions.json', 'r') as f:
    all_questions = json.load(f)

# Load your existing training data to exclude those questions
with open('training_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    used_questions = set(row['question'] for row in reader)

# Filter out anything already used in training
unused = [q for q in all_questions if q['question'] not in used_questions]
print(f"Available unused questions: {len(unused)}")

# Sample 60 fresh ones — different seed than before
random.seed(123)
test_sample = random.sample(unused, 60)

with open('test_set.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['company', 'ticker', 'exchange', 'date', 'analyst_name', 'question', 'label'])
    writer.writeheader()
    for q in test_sample:
        q['label'] = ''
        writer.writerow(q)

print("Saved 60 questions to test_set.csv — label these manually")