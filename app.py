import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("BeratCan08/VehicleGPT2")
    tokenizer = GPT2Tokenizer.from_pretrained("BeratCan08/VehicleGPT2")
    tokenizer.pad_token = tokenizer.eos_token  # Önemli
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

model, tokenizer = load_model()

st.title("🚗 Vehicle Chatbot")
user_input = st.text_input("Sorunuzu yazınız:")

if user_input:
    prompt = user_input.strip() + " "
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=inputs["input_ids"].shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    st.write("Cevap:", response)
