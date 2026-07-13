import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 🔮 Arayüz ayarları
st.set_page_config(page_title="AlphaAI", page_icon="🔮")
st.title("AlphaAI")

# 🛠️ TAM DOSYA ADI: Senin o dosyandaki boşlukları tırnak içinde aynen kopyaladım.
# API %20 aramıyor, senin o boşluk karakterlerini arıyor.
REPO_ID = "walkmane/AlphaAI"
MODEL_FILE = "ALPHA-1.3-3b- stable_version Q4_K_M.gguf"

@st.cache_resource
def load_model():
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        # Dosyayı indir
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=MODEL_FILE,
            token=hf_token
        )
        
        # CPU için optimize edilmiş model ayarları
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,        
            n_threads=4,       
            n_gpu_layers=0,    # GPU kapatıldı, %100 CPU
            n_batch=512,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        return None

llm = load_model()

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesajları göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Mesajını yaz..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor..."):
            llama_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            try:
                if llm:
                    response = llm(
                        llama_prompt, 
                        max_tokens=256, 
                        temperature=0.7, 
                        top_p=0.9, 
                        repeat_penalty=1.1,
                        stop=["<|eot_id|>"], 
                        echo=False
                    )
                    answer = response["choices"][0]["text"].strip()
                else:
                    answer = "Model yüklenemedi, logları kontrol et kankooo."
            except Exception as e:
                answer = f"Hata: {str(e)}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
