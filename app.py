import os
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.set_page_config(page_title="AlphaAI", page_icon="🔮")
st.title("AlphaAI")

REPO_ID = "walkmane/AlphaAI"
MODEL_FILE = "ALPHA-1.3-3b- stable_version Q4_K_M.gguf"

# 🛑 CHARACTER SETTINGS ARE FIXED INSIDE THE CODE
SYSTEM_PROMPT = "You are the CakeLLM language model trained by Alpha AI; your task is to assist. "

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

# Keep the chat history clean
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 🔗 Combine the prompt with the system instruction
        llama_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        try:
            if llm:
                stream = llm(
                    llama_prompt,
                    max_tokens=150,
                    temperature=0.7,
                    stream=True
                )
                for chunk in stream:
                    content = chunk["choices"][0]["text"]
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                message_placeholder.markdown("Failed to load the model.")
        except Exception as e:
            message_placeholder.markdown("The server is under heavy load. Please try again!")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
