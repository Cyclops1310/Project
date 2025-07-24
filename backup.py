import os
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# === Load model ===
model_id = "meta-llama/Llama-3.2-1B-Instruct"
print(f"🔄 Loading model: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device set to use: {device}")

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    return_full_text=False
)

# === Session directory ===
CHAT_DIR = "chat"
os.makedirs(CHAT_DIR, exist_ok=True)

# === Session selection ===
while True:
    choice = input("Start new chat or continue old? (new/old): ").strip().lower()

    if choice == "new":
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_file = os.path.join(CHAT_DIR, f"{session_id}.json")
        print(f"🆕 New chat started with session ID: {session_id}")
        chat_history = []
        break

    elif choice == "old":
        session_id = input("Enter previous session ID (e.g., 20250722_153000): ").strip()
        chat_file = os.path.join(CHAT_DIR, f"{session_id}.json")
        if os.path.exists(chat_file):
            print(f"🔁 Resuming chat session: {session_id}")
            with open(chat_file, "r") as f:
                chat_history = json.load(f)
            break
        else:
            print("❌ Session not found. Try again.\n")

    else:
        print("❗ Invalid choice. Please type 'new' or 'old'.")

# === Save chat function ===
def save_chat():
    with open(chat_file, "w") as f:
        json.dump(chat_history, f, indent=2)

# === Prompt builder ===
def build_prompt(history):
    prompt = "System: You are a helpful assistant.\n"
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant:"  # continue from last user message
    return prompt

# === Clean response ===
def clean_response(text):
    for stop in ["User:", "You:", "Human:", "Assistant:"]:
        if stop in text:
            return text.split(stop)[0].strip()
    return text.strip()

# === Chat loop ===
print("\n🤖 Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Chat ended.")
        break

    # Save user message
    chat_history.append({"role": "user", "content": user_input})

    # Build prompt from full history
    prompt = build_prompt(chat_history)
    print("\n🔍 Prompt to model:\n", prompt)  # DEBUG — optional

    try:
        result = pipe(prompt)
        raw_bot_reply = result[0]["generated_text"]
        generated_part = raw_bot_reply.strip()  # already excludes prompt since return_full_text=False
        bot_reply = clean_response(generated_part)

        print(f"Bot: {bot_reply}\n")
        chat_history.append({"role": "assistant", "content": bot_reply})
        save_chat()
    except Exception as e:
        print(f"⚠️ Error: {e}")
