import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
import torch
from trl import SFTTrainer, SFTConfig

# Load your training data
df = pd.read_csv("training_data.csv")
print(f"Loaded {len(df)} training examples")
print(df['label'].value_counts())

# Format into instruction tuning format
def format_example(row):
    return {
        "text": f"""Below is an analyst question from an earnings call. Classify the sentiment tone as one of: Confident, Cautious, Hedging, or Alarmed. Explain your reasoning in one sentence.

### Question:
{row['question']}

### Classification:
{row['label']} — """
    }

formatted = df.apply(format_example, axis=1).tolist()
dataset = Dataset.from_list(formatted)
print(f"Dataset ready: {len(dataset)} examples")

# Load tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA config — bfloat16 to match training precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model loaded successfully")

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

sft_config = SFTConfig(
    output_dir="./toneshift-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    report_to="none",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=sft_config,
)

print("Starting training...")
trainer.train()
print("Training complete!")

trainer.model.save_pretrained("./toneshift-lora")
tokenizer.save_pretrained("./toneshift-lora")
print("Model saved to ./toneshift-lora")
