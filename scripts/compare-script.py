#this script compares our fine tune mistral 7B with the raw mistral 7B, finding out the effects of our fine-tuning
import torch, csv, re, json
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LABELS = ["Confident", "Cautious", "Hedging", "Alarmed"]

with open("test_set_Labelled.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Test set: {len(rows)} questions")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")

def prompt_for(q):
    return f"""Below is an analyst question from an earnings call. Classify the sentiment tone as one of: Confident, Cautious, Hedging, or Alarmed. Explain your reasoning in one sentence.

### Question:
{q}

### Classification:
"""

def parse(text):
    """Pull the first valid label out of the generation. None = unparseable."""
    for label in LABELS:
        if re.search(rf"\b{label}\b", text, re.I):
            return label
    return None

def run(model, tag):
    preds = []
    for i, r in enumerate(rows):
        inputs = tokenizer(prompt_for(r["question"]), return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append({"true": r["label"].strip(), "pred": parse(gen), "raw": gen.strip(),
                      "question": r["question"][:100]})
        if (i + 1) % 10 == 0:
            print(f"  {tag}: {i+1}/{len(rows)}")
    return preds

def report(preds, tag):
    n = len(preds)
    correct = sum(p["true"] == p["pred"] for p in preds)
    unparseable = sum(p["pred"] is None for p in preds)
    print(f"\n=== {tag} ===")
    print(f"Accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"Unparseable outputs: {unparseable}")

    print("\nPer-class:")
    for label in LABELS:
        subset = [p for p in preds if p["true"] == label]
        if subset:
            hit = sum(p["pred"] == label for p in subset)
            print(f"  {label:10s} {hit}/{len(subset)} = {hit/len(subset):.0%}")

    print("\nConfusion (true → predicted):")
    conf = defaultdict(Counter)
    for p in preds:
        conf[p["true"]][p["pred"] or "UNPARSEABLE"] += 1
    for label in LABELS:
        if conf[label]:
            print(f"  {label:10s} → {dict(conf[label])}")
    return correct / n

print("\nRunning BASE model...")
base_preds = run(base, "base")
base_acc = report(base_preds, "BASE Mistral 7B")

print("\nRunning FINE-TUNED model...")
tuned = PeftModel.from_pretrained(base, "./toneshift-lora")
tuned.eval()
tuned_preds = run(tuned, "tuned")
tuned_acc = report(tuned_preds, "FINE-TUNED ToneShift")

print(f"\n{'='*40}")
print(f"Base:       {base_acc:.1%}")
print(f"Fine-tuned: {tuned_acc:.1%}")
print(f"Delta:      {tuned_acc - base_acc:+.1%}")

with open("eval_results.json", "w") as f:
    json.dump({"base_accuracy": base_acc, "tuned_accuracy": tuned_acc,
               "base_preds": base_preds, "tuned_preds": tuned_preds}, f, indent=2)
print("\nSaved eval_results.json")