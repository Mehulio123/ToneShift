#Python script for parsing the data, and getting just the calls seperately from it!
from datasets import load_dataset # importing the data set

ds = load_dataset("kunhanw/earning_call_transcript_dataset_with_volatility_analysis",
    cache_dir="C:/Projects/ToneShift/data/raw") # loading the data set

text = ds['train'][0]['text'] # getting the text from the first entry in the training set
meta = ds['train'][0]['meta'] # getting the metadata from the first entry in the training set
qa_start = text.find('Question-and-Answer Session') # finding the start of the Q&A session in the text
if qa_start == -1:
    qa_start = text.find('Q&A')

print('Company:', meta['company']) # printing the company name from the metadata
print('QA section starts at character:', qa_start) # printing the character index where the Q&A section starts
print()
print('QA SECTION:')
print(text[qa_start:qa_start+5000]) # printing the first 2000 characters of the Q&A section to get a sense of its content

