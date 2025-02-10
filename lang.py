import streamlit as st
from langchain_community.llms import HuggingFaceHub


# Set up Streamlit UI
st.title("LangChain + Hugging Face Model in Streamlit")
st.write("Enter a prompt, and the model will generate a response.")

# User input
user_input = st.text_area("Enter your prompt:")

# Load the model from Hugging Face Hub
llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.7, "max_length": 100})

# Generate response
if st.button("Generate Response"):
    if user_input:
        response = llm(user_input)
        st.subheader("Generated Response:")
        st.write(response)
    else:
        st.write("Please enter a prompt before generating a response.")
