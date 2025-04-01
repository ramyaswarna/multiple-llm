import streamlit as st
import requests

API_URL = "https://multiple-llm-backend.onrender.com"  # Update this after backend deploy

models = {
    "Qwen/Qwen2-0.5B": "Qwen2-0.5B (500M parameters)",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B (1.5B parameters)",
    "Qwen/Qwen1.5-1.8B": "Qwen1.5-1.8B (1.8B parameters)",
    "EleutherAI/gpt-neo-1.3B": "GPT-Neo (1.3B parameters)"
}

st.title("Interactive Chatbot (via API)")

selected_model = st.selectbox("Select a model:", options=list(models.keys()), format_func=lambda x: models[x])
user_question = st.text_area("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Contacting backend..."):
            res = requests.post(API_URL, json={"question": user_question, "model_name": selected_model})
            if res.status_code == 200:
                st.subheader("Answer:")
                st.write(res.json()["answer"])
            else:
                st.error("Backend error. Please try again.")
    else:
        st.warning("Please enter a question.")
