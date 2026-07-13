import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

# 🔮 Arayüz ayarları
st.set_page_config(page_title="AlphaAI", page_icon="🔮")
st.title("AlphaAI")

REPO_ID = "walkmane/AlphaAI"
MODEL_FILE = "ALPHA-1.3-3b- stable_version Q4_K_M.gguf"

@st.cache_resource
def load_model():
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=MODEL_FILE,
            token=hf_token
        )
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,        
            n_threads=4,       
            n_gpu_layers=0,    
            n_batch=512,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        return None

llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Mesajını yaz..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Canlı akış için stream=True kullanıyoruz
        llama_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        try:
            if llm:
                # stream=True ile kelime kelime alıyoruz
                stream = llm(
                    llama_prompt, 
                    max_tokens=256, 
                    temperature=0.7, 
                    top_p=0.9, 
                    repeat_penalty=1.1,
                    stop=["<|eot_id|>"], 
                    echo=False,
                    stream=True 
                )
                
                for chunk in stream:
                    content = chunk["choices"][0]["text"]
                    full_response += content
                    # Canlı yazma efekti
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                message_placeholder.markdown("Model yüklenemedi.")
        except Exception as e:
            message_placeholder.markdown(f"Hata: {str(e)}")
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
