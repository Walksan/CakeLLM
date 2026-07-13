import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 🔮 Sayfa başlığı ve tasarımı ayarlanıyor ("Terminal" kelimesi yok!)
st.set_page_config(page_title="AlphaAI Arayüzü", page_icon="🔮", layout="centered")
st.title("🔮 AlphaAI Arayüzü (ALPHA-1.3-3b-stable)")
st.caption("Tamamen CPU üzerinde çalışan, kararlı ve emoji spamsız CakeLLM sistemi! 💃🏼🛸")

# 📂 Model bilgileri ve main branch'indeki boşluklu stabil sürüm adı
REPO_ID = "walkmane/AlphaAI"
MODEL_FILE = "ALPHA-1.3-3b-%20stable_version%20Q4_K_M.gguf "

# Modeli sadece bir kez yüklemek için önbelleğe (cache) alıyoruz
@st.cache_resource
def load_model():
    # Model depon gizliyse (Private) Streamlit Secrets'tan tokenı çeker
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=MODEL_FILE,
            token=hf_token
        )
        # Sadece CPU motoru çalışacak şekilde Llama kurulumu
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,        
            n_threads=4,       
            n_gpu_layers=0     
        )
        return llm
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

llm = load_model()

# Sohbet geçmişini Streamlit hafızasında başlatıyoruz
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana basıyoruz
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcıdan girdi alma alanı
if prompt := st.chat_input("AlphaAI'a bir şeyler yaz..."):
    # Kullanıcı mesajını ekrana bas ve hafızaya ekle
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Modelin cevap üretme aşaması
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor... ⏳"):
            # Temiz Llama-3 prompt şablonu
            llama_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            try:
                response = llm(
                    llama_prompt,
                    max_tokens=256,   # CPU hızı için dengeli token sınırı
                    temperature=0.7, 
                    top_p=0.9,
                    stop=["<|eot_id|>", "Kullanıcı:"], 
                    echo=False
                )
                answer = response["choices"][0]["text"].strip()
            except Exception as e:
                answer = f"Aykana kodlar şizil-buzul oldu! 💥 Hata: {str(e)}"
            
            st.markdown(answer)
            
    # Asistanın cevabını hafızaya ekle
    st.session_state.messages.append({"role": "assistant", "content": answer})

