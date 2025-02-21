import torch
torch.mps.empty_cache()  # Clears GPU memory
import streamlit as st
from transformers import pipeline
import torch

# Force PyTorch to use CPU instead of MPS (Mac GPU)
torch.device("cpu")

# Load a smaller, free model with CPU mode
generator = pipeline("text-generation", model="Salesforce/codegen-350M-multi", device=-1)  # -1 forces CPU

def generate_code(prompt):
    full_prompt = f"Write a program: {prompt}"
    
    torch.mps.empty_cache()  # Clear GPU memory before running
    response = generator(full_prompt, max_length=5000, do_sample=True, temperature=0.7)
    
    return response[0]['generated_text']

# Streamlit UI
st.title("TARS AI CODE GENERATOR")

# User input
prompt = st.text_area("Enter your code request", "Write a Python function to check prime numbers")

# Generate code button
if st.button("Building Code:"):
    torch.mps.empty_cache()  # Free GPU memory
    generated_code = generate_code(prompt)
    st.code(generated_code, language="python")  # Syntax highlighting
