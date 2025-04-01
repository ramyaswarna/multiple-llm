# backend/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = FastAPI()

models = {
    "Qwen/Qwen2-0.5B": "Qwen2-0.5B (500M parameters)",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B (1.5B parameters)",
    "Qwen/Qwen1.5-1.8B": "Qwen1.5-1.8B (1.8B parameters)",
    "EleutherAI/gpt-neo-1.3B": "GPT-Neo (1.3B parameters)"
}

# Pre-load models into a dictionary for faster switching (optional)
loaded_models = {}

class Query(BaseModel):
    question: str
    model_name: str

def get_model(model_name):
    if model_name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template="You are an AI assistant. Answer the following question concisely and accurately:\n\nQuestion: {question}\nAnswer:"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        loaded_models[model_name] = chain
    return loaded_models[model_name]

@app.post("/generate")
def generate_answer(query: Query):
    if query.model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    chain = get_model(query.model_name)
    answer = chain.run(question=query.question)
    return {"answer": answer.strip()}
