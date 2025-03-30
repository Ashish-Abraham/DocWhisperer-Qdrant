import os
from pathlib import Path
import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.settings import Settings
import ingest
from load_model import load_models

st.set_page_config(
    page_title="DocWhisperer", 
    page_icon="🧙‍♂️", 
    layout="wide", 
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #9933FF; 
        color: white; 
    }
    .stTextInput>div>div>input { border-radius: 5px; }
    .css-1d391kg { padding: 2rem 1rem; }
    .stAlert { padding: 1rem; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_vector_store():
    client = QdrantClient(host="localhost", port=6333)
    return QdrantVectorStore(client=client, collection_name="ASM")

@st.cache_resource
def load_data(embed_model, llm):
    vector_store = initialize_vector_store()
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    index = VectorStoreIndex.from_vector_store(vector_store, service_context)
    return index

def main():
    st.sidebar.image("images/w1_noback.png", width=100)
    st.sidebar.title("DocWhisperer")
    st.sidebar.markdown("---")

    uploaded_files = st.sidebar.file_uploader(
        "📄 Upload your PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing your documents... 📚"):
            try:
                client = QdrantClient(host="localhost", port=6333)
                ingest.ingest_pdfs(uploaded_files, client, "ASM")
                st.success("✨ Documents successfully processed!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    embed_model, llm = load_models()
    Settings.embed_model = embed_model

    st.title("DocWhisperer 🧙‍♂️")

    if not st.session_state.messages:
        st.markdown("""
        ### 👋 Welcome to DocWhisperer!

        I'm your magical document assistant. Here's how to get started:
        1. Upload your PDF documents using the sidebar
        2. Ask me anything about your documents
        3. I'll search through them and provide detailed answers

        Let's begin our journey through your documents! ✨
        """)

    index = load_data(embed_model, llm)
    chat_engine = index.as_chat_engine(streaming=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me anything about your documents... 💭"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = chat_engine.stream_chat(prompt)
                    response_text = st.write_stream(response.response_gen)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error("I apologize, but I encountered an error. Please try rephrasing your question.")

if __name__ == '__main__':
    main()