import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 🔮 Page Config
st.set_page_config(page_title="CakeLLM", page_icon="🍰")

# 🌍 Dil Seçimi
lang = st.sidebar.selectbox("Select Language / Dil Seçin", ["English", "Türkçe"])

# Dil sözlüğü
texts = {
    "English": {"title": "🍰 CakeLLM Assistant", "input": "Type your message...", "working": "Working on it...", "error": "Model could not be loaded.", "server_err": "Server is under heavy load, please try again!"},
    "Türkçe": {"title": "🍰 CakeLLM Asistanı", "input": "Mesajını yaz...", "working": "Çalışıyor...", "error": "Model yüklenemedi.", "server_err": "Sunucu çok zorlandı, tekrar dene!"}
}

st.title(texts[lang]["title"])

REPO_ID = "walkmane/AlphaAI"
MODEL_FILE = "ALPHA-1.3-3b- stable_version Q4_K_M.gguf"

# 🛑 SYSTEM PROMPT (Değişmez)
SYSTEM_PROMPT = "You are the CakeLLM language model trained by Alpha AI; your task is to assist. Be realistic, friendly, and natural. Use emojis sparingly. Keep responses concise and avoid robotic language."

@st.cache_resource
def load_model():
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, token=hf_token)
        llm = Llama(
            model_path=model_path,
            n_ctx=1024,        
            n_threads=2,       
            n_gpu_layers=0,    
            n_batch=128,       
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(texts[lang]["input"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "assistant"
            full_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        try:
            if llm:
                with st.spinner(texts[lang]["working"]):
                    stream = llm(full_prompt, max_tokens=150, temperature=0.7, stream=True)
                    for chunk in stream:
                        full_response += chunk["choices"][0]["text"]
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            else:
                message_placeholder.markdown(texts[lang]["error"])
        except Exception:
            message_placeholder.markdown(texts[lang]["server_err"])
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
