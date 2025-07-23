import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    outputs = pipe(
        user_input,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    response = outputs[0]["generated_text"][len(user_input):].strip()
    print(f"Bot: {response}")
