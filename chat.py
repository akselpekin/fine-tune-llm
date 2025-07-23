import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_NAME = "./results/checkpoint-0000"  # CHANGE TO LATEST CHECKPOINT BEFORE USE !!!

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("Type 'exit' to quit.")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=64,
            do_sample=True,
            top_k=40,
            top_p=0.92,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Model: {response.strip()}\n")