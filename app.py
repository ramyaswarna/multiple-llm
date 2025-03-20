import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

models = {
    "Qwen/Qwen2-0.5B": "Qwen2-0.5B (500M parameters)",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B (1.5B parameters)",
    "Qwen/Qwen1.5-1.8B": "Qwen1.5-1.8B (1.8B parameters)",
    "EleutherAI/gpt-neo-1.3B": "GPT-Neo (1.3B parameters)"
}

st.title("Interactive Chatbot")

# Model selection
selected_model = st.selectbox("Select a model:", options=list(models.keys()), format_func=lambda x: models[x])

# Cache model loading
@st.cache_resource
def load_model(model_name):
    """Load the specified model and tokenizer for CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return HuggingFacePipeline(pipeline=text_gen_pipeline)

# Load the selected model
with st.spinner("Loading model..."):
    llm = load_model(selected_model)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are an AI assistant. Answer the following question concisely and accurately:\n\nQuestion: {question}\nAnswer:"
)

# Create the LangChain LLMChain
qa_chain = LLMChain(llm=llm, prompt=prompt_template)

# User input
user_question = st.text_area("Enter your question:")

# Generate answer
if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Generating answer..."):
            try:
                answer = qa_chain.run(question=user_question)
                st.subheader("Answer:")
                st.write(answer.strip())
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.write("Powered by LangChain and Hugging Face Transformers (CPU Optimized).")
