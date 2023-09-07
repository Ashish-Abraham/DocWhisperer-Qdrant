import streamlit as st
from transformers import Autotokenizer, pipeline, AutoModelforSeq2SeqLM
import torch
import base64
import textwrap

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms  import HuggingFacePipeline
from constants import CHROMA_SETTINGS

checkpoint=""
tokenizer = Autotokenizer.from_pretrained(checkpoint)
base_model = AutoModelforSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe=pipeline(
        'text-2-text generation',
        model=base_model,
        tokenizer=tokenizer,
        temperature=0.4, 
        max_length=256,
        do_sample=True,
        top_p=0.95
    )
    local_llm=HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm=llm_pipeline()
    embeddings=SentenceTransformerEmbeddings(model_name="")
    db=Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever=db.as_retriever()
    qa=RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

