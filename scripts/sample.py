#list to randomly pick 500 questions that I can label mannually for fine tuning my model 
import json
import random
import csv

with open('data/analyst_questions.json', 'r') as f:
    all_questions = json.load(f)

random.seed(42)
sample = random.sample(all_questions, 500)

with open('labeling_sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['company', 'ticker', 'exchange', 'date', 'analyst_name', 'question', 'label'])
    writer.writeheader()
    for q in sample:
        q['label'] = ''
        writer.writerow(q)

print("Saved 500 questions to labeling_sample.csv")