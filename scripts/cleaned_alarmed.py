import csv

# These are management names that snuck through
management_names = {
    "Marianne Lake", "Derek Dewan", "Eric Langan", "Daniel Ek",
    "Bradley Chhay", "Jeffery Cathey", "Amy Bunszel", "Simon Mays-Smith",
    "Doug Bettinger", "Bryan Giglia", "A - Leslie Garber",
    "A - Derek Dewan", "Unidentified Company Representative"
}

with open('alarmed_candidates.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter out management responses
cleaned = []
removed = []

for row in rows:
    analyst = row['analyst_name'].strip()
    question = row['question'].strip()
    
    # Remove if name is in management list
    if analyst in management_names:
        removed.append(row)
        continue
    
    # Remove if text doesn't contain a question mark (likely an answer not a question)
    if '?' not in question:
        removed.append(row)
        continue
    
    cleaned.append(row)

print(f"Original: {len(rows)}")
print(f"Kept: {len(cleaned)}")
print(f"Removed: {len(removed)}")

# Save cleaned version
with open('alarmed_cleaned.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['company', 'ticker', 'exchange', 'date', 'analyst_name', 'question', 'label'])
    writer.writeheader()
    writer.writerows(cleaned)

print("Saved to alarmed_cleaned.csv")