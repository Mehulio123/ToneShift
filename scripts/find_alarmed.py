import json
import csv

with open('./data/analyst_questions.json', 'r') as f:
    questions = json.load(f)

# Keywords associated with Alarmed tone
alarmed_keywords = [
    "missed", "why didn't", "below guidance", "disappointed",
    "shortfall", "concern", "failed", "why did", "fell short",
    "below expectations", "weak", "struggle", "deteriorat",
    "why haven't", "not meeting", "underperform", "loss",
    "declining", "worrying", "alarming", "cut guidance",
    "lowered", "revised down", "writedown", "impairment",
    "why is", "how did you miss", "what went wrong"
]

# Score each question by how many keywords it contains
scored = []
for q in questions:
    text = q['question'].lower()
    score = sum(1 for keyword in alarmed_keywords if keyword in text)
    if score > 0:
        scored.append((score, q))

# Sort by score descending
scored.sort(key=lambda x: x[0], reverse=True)

# Take top 150
top_alarmed = [q for score, q in scored[:150]]

print(f"Found {len(scored)} questions with alarmed keywords")
print(f"Saving top 150 candidates...")

with open('alarmed_candidates.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['company', 'ticker', 'exchange', 'date', 'analyst_name', 'question', 'label'])
    writer.writeheader()
    for q in top_alarmed:
        q['label'] = 'Alarmed'
        writer.writerow(q)

print("Saved to alarmed_candidates.csv")
print("Open it, verify the labels look right, then we'll merge with your existing labels")