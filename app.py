import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Araç Chatbot", page_icon="🚗")
st.title("🚗 Araç Chatbot")
st.write("Araçlarla ilgili sorularınızı cevaplıyorum!")

@st.cache_resource
def load_model():
    model_id = "BeratCan08/VehicleGPT2"  # kendi Hugging Face model adını yaz
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Soru:", "")

if st.button("Gönder") and user_input:
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.chat_history.append(("Sen", user_input))
    st.session_state.chat_history.append(("Bot", response))

for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")
