import json
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model_name = "openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

dataset = load_dataset("json", data_files="data.jsonl")["train"]

def preprocess(example):
    text = f"{example['prompt']} {tokenizer.eos_token} {example['response']}{tokenizer.eos_token}"
    result = tokenizer(text, truncation=True, padding="max_length", max_length=64)
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=5,
    save_total_limit=2,
    logging_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()

prompt = "What do you know about Aksel Pekin?"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids).to(device)
model.to(device)
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=20,
    do_sample=False,
)
print(tokenizer.decode(output[0]))