import json
import csv

with open('data/analyst_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

with open('training_data.csv', 'r', encoding='utf-8') as f:
    train_qs = set(r['question'].strip() for r in csv.DictReader(f))

with open('test_set_Labelled.csv', 'r', encoding='utf-8') as f:
    test_qs = set(r['question'].strip() for r in csv.DictReader(f))

used_qs = train_qs | test_qs

# Keyword lists are just a search heuristic to surface likely candidates.
# They are NOT ground truth - every row below still needs a human to read
# it and assign (or reject) a label. A question can mention a bad number
# in a calm/cautious tone, or a good number in a neutral tone, so keyword
# hits are noisy and should not be trusted as labels on their own.
ALARMED_KEYWORDS = [
    "missed", "why didn't", "below guidance", "disappointed",
    "shortfall", "concern", "failed", "why did", "fell short",
    "below expectations", "weak", "struggle", "deteriorat",
    "why haven't", "not meeting", "underperform", "loss",
    "declining", "worrying", "alarming", "cut guidance",
    "lowered", "revised down", "writedown", "impairment",
    "why is", "how did you miss", "what went wrong",
]

CONFIDENT_KEYWORDS = [
    "great quarter", "impressive", "strong execution", "beat",
    "raised guidance", "ahead of expectations", "outperform",
    "record", "robust", "accelerat", "tailwind", "well positioned",
    "congrats", "congratulations", "nice job", "solid results",
    "exceeded", "upside", "momentum", "encouraging", "kudos",
    "well done", "great job", "strong quarter",
]

HEDGING_KEYWORDS = [
    "i guess", "sort of", "kind of", "not sure if", "wondering if",
    "just curious", "how should we think about", "any color on",
    "just to clarify", "correct me if i'm wrong", "i may be wrong",
    "it seems like", "to some extent", "depends on", "we'll see",
    "hard to say", "unclear", "i don't know if", "if i'm not mistaken",
    "just want to make sure", "would you say", "is it fair to say",
    "i'm just trying to understand", "not entirely clear",
]

def score(text, keywords):
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)

candidates = []
for q in questions:
    text = q['question'].strip()
    if text in used_qs:
        continue
    a_score = score(text, ALARMED_KEYWORDS)
    c_score = score(text, CONFIDENT_KEYWORDS)
    h_score = score(text, HEDGING_KEYWORDS)
    best = max(a_score, c_score, h_score)
    if best == 0:
        continue
    if best == a_score:
        candidates.append((a_score, 'Alarmed', q))
    elif best == c_score:
        candidates.append((c_score, 'Confident', q))
    else:
        candidates.append((h_score, 'Hedging', q))

alarmed = sorted([c for c in candidates if c[1] == 'Alarmed'], key=lambda x: -x[0])
confident = sorted([c for c in candidates if c[1] == 'Confident'], key=lambda x: -x[0])
hedging = sorted([c for c in candidates if c[1] == 'Hedging'], key=lambda x: -x[0])

POOL_SIZE = 150
top = alarmed[:POOL_SIZE] + confident[:POOL_SIZE] + hedging[:POOL_SIZE]

with open('candidates_to_label.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['company', 'ticker', 'exchange', 'date', 'analyst_name',
                    'question', 'suggested_label', 'label'],
    )
    writer.writeheader()
    for keyword_score, suggested, q in top:
        writer.writerow({
            'company': q['company'],
            'ticker': q['ticker'],
            'exchange': q['exchange'],
            'date': q['date'],
            'analyst_name': q['analyst_name'],
            'question': q['question'],
            'suggested_label': suggested,
            'label': '',
        })

print(f"Alarmed-leaning candidates found: {len(alarmed)} (saved top {min(POOL_SIZE, len(alarmed))})")
print(f"Confident-leaning candidates found: {len(confident)} (saved top {min(POOL_SIZE, len(confident))})")
print(f"Hedging-leaning candidates found: {len(hedging)} (saved top {min(POOL_SIZE, len(hedging))})")
print(f"Total rows in candidates_to_label.csv: {len(top)}")
print()
print("Next: open candidates_to_label.csv. For each row, read the question and")
print("fill in the 'label' column yourself with Confident/Cautious/Hedging/Alarmed")
print("('suggested_label' is only a hint - ignore it if the question doesn't")
print("actually read that way). Leave 'label' blank on rows you don't want to")
print("keep. Then run merge_test_candidates.py.")
