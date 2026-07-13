import csv
from collections import Counter

# Load original 500 labels
with open('Labelled-Sample.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    original = list(reader)

# Load cleaned alarmed examples
with open('alarmed_cleaned.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    alarmed = list(reader)

# Combine
merged = original + alarmed

# Save
with open('training_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['company', 'ticker', 'exchange', 'date', 'analyst_name', 'question', 'label'])
    writer.writeheader()
    writer.writerows(merged)

labels = [row['label'].strip() for row in merged]
counts = Counter(labels)

print(f"Total training examples: {len(merged)}")
print("\nLabel distribution:")
for label, count in counts.most_common():
    print(f"  {label}: {count}")