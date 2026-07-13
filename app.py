import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 🔮 Arayüz başlık ayarları
st.set_page_config(page_title="AlphaAI Arayüzü", page_icon="🔮", layout="centered")
st.title("🔮 AlphaAI Arayüzü (ALPHA-1.3-3b-stable)")

# 🛠️ TAM DOSYA ADI: Senin belirttiğin `%20` kodlamasıyla birebir aynı!
REPO_ID = "walkmane/AlphaAI"
MODEL_FILE = "ALPHA-1.3-3b-%20stable_version%20Q4_K_M.gguf"

@st.cache_resource
def load_model():
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        # Sunucunun dosyayı %20 karakterleriyle bulması için tam isimle çekiyoruz
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=MODEL_FILE,
            token=hf_token
        )
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,        
            n_threads=4,       
            n_gpu_layers=0     
        )
        return llm
    except Exception as e:
        st.error(f"Dosya adı veya token hatası! Sunucu şu ismi bulamıyor: {MODEL_FILE}. Hata: {str(e)}")
        return None

llm = load_model()

# Sohbet arayüzü
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("AlphaAI'a bir şeyler yaz..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor... ⏳"):
            llama_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            try:
                if llm is not None:
                    response = llm(
                        llama_prompt,
                        max_tokens=256,
                        temperature=0.7, 
                        top_p=0.9,
                        stop=["<|eot_id|>", "Kullanıcı:"], 
                        echo=False
                    )
                    answer = response["choices"][0]["text"].strip()
                else:
                    answer = "Model yüklenemedi."
            except Exception as e:
                answer = f"Kod hatası: {str(e)}"
            
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
