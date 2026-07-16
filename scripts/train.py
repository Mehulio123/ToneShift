import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Attach your fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, "./toneshift-lora")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./toneshift-lora")

# Test
test_question = "Can you help us understand why margins didn't hit the guidance you gave last quarter? This is the third consecutive miss."

prompt = f"""Below is an analyst question from an earnings call. Classify the sentiment tone as one of: Confident, Cautious, Hedging, or Alarmed. Explain your reasoning in one sentence.

### Question:
{test_question}

### Classification:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Classification:")[1])