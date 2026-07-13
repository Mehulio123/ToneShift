#script to list out top 30 companies
import json
from collections import Counter

with open('data/analyst_questions.json', 'r') as f:
    questions = json.load(f)

companies = Counter(q['ticker'] for q in questions)
print("Top 30 companies by question count:")
for company, count in companies.most_common(30):
    print(f"{company}: {count} questions")
