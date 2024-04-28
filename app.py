import datetime
import os
import random
from pathlib import Path
from typing import Any
import pandas as pd
import datasets
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader, ServiceContext)
from llama_index.core.settings import Settings
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
import streamlit as st
import ingest  # Import the ingest.py file
from load_model import load_models  # Import the load_models function

st.set_page_config(page_title="DocWhisperer", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
client = QdrantClient(host='localhost', port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="ASM")

@st.cache_resource(show_spinner=False)
def load_data(_llm):
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

# Add a file uploader for PDF files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Invoke ingest.py with the uploaded PDF files
    ingest.ingest_pdfs(uploaded_files, client, "ASM")

if __name__ == '__main__':
    embed_model, llm = load_models()  # Load the models
    Settings.llm = llm
    Settings.embed_model = embed_model
    st.title("DocWhisperer🧙‍♂️")  # Changed the title
    index = load_data(llm)
    if "chat_engine" not in st.session_state.keys():
        # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    if prompt := st.chat_input("Your question"):
        # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        for message in st.session_state.messages:
            # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)
