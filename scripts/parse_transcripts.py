import re
import json
from datasets import load_dataset

ds = load_dataset("kunhanw/earning_call_transcript_dataset_with_volatility_analysis",
                  cache_dir="C:/Projects/ToneShift/data/raw")
def parse_analyst_questions(text, meta):
    questions = []
    qa_start = text.find('Question-and-Answer Session')
    if qa_start == -1:
        return questions  # No Q&A section found
    
    qa_text = text[qa_start:] # Extract the Q&A section

    # Get company participants (management) to exclude them
    participants_start = text.find('Company Participants')
    participants_end = text.find('Conference Call Participants')
    management_names = set()
    
    if participants_start != -1 and participants_end != -1:
        mgmt_section = text[participants_start:participants_end]
        for line in mgmt_section.split('\n'):
            line = line.strip()
            if ' - ' in line:
                name = line.split(' - ')[0].strip()
                management_names.add(name)
    management_names.add('Operator')  # Add 'Operator' to the set of management names

    #split into lanes and find analyst questions
    lines = qa_text.split('\n')
    lines = [l.strip() for l in lines if l.strip()] # remove empty lines and strip whitespace

    i = 0

    while i < len(lines):
        line = lines[i]
        
        # Check if this line looks like a speaker name
        # Names are short, no punctuation, not a question/sentence
        is_name = (
            len(line.split()) <= 4 and
            not line.endswith('?') and
            not line.endswith('.') and
            len(line) < 50 and
            not line.startswith('[')
        )
        
        if is_name and line not in management_names:
            # Next lines are their question
            question_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # Stop when we hit another name
                is_next_name = (
                    len(next_line.split()) <= 4 and
                    not next_line.endswith('?') and
                    not next_line.endswith('.') and
                    len(next_line) < 50 and
                    not next_line.startswith('[')
                )
                if is_next_name:
                    break
                question_lines.append(next_line)
                j += 1
            
            question_text = ' '.join(question_lines).strip()
            
            if question_text and len(question_text) > 50:
                questions.append({
                    'company': meta.get('company', ''),
                    'ticker': meta.get('name', ''),
                    'exchange': meta.get('exchange', ''),
                    'date': meta.get('publishOn', '')[:10],
                    'analyst_name': line,
                    'question': question_text
                })
            
            i = j
        else:
            i += 1
    
    return questions

# Test on first 5 transcripts
all_questions = []
for i in range(len(ds['train'])):
    text = ds['train'][i]['text']
    meta = ds['train'][i]['meta']
    questions = parse_analyst_questions(text, meta)
    all_questions.extend(questions)
    if i % 100 == 0:
        print(f"Processed {i}/1831 transcripts... ({len(all_questions)} questions so far)")

print(f"\nDone. Total questions extracted: {len(all_questions)}")   

import json 
with open('analyst_questions.json', 'w') as f:
    json.dump(all_questions, f, indent=2)

print("saved to data/questions.json")