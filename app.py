import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Modeli ve tokenizer'ı yükle
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("BeratCan08/VehicleGPT2")
    tokenizer = GPT2Tokenizer.from_pretrained("BeratCan08/VehicleGPT2")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit arayüzü
st.title("🚗 Vehicle Chatbot")
st.write("Araçlarla ilgili sorularınızı sorun!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

user_input = st.text_input("Siz:", key="input")

if user_input:
    prompt = st.session_state.chat_history + f"User: {user_input}\nBot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Uyarı için maske ekliyoruz

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=inputs.shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=1
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = output_text[len(prompt):].strip().split("\n")[0]

    st.session_state.chat_history += f"User: {user_input}\nBot: {response}\n"
    st.text_area("Sohbet:", st.session_state.chat_history, height=300)

