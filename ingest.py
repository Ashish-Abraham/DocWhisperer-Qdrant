import os
from pathlib import Path
from llama_index import GPTVectorStoreIndex, GPTTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.qdrant import QdrantVectorStore

def ingest_pdfs(uploaded_files, client, collection_name):
    documents = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file.read())
        docs = loader.load()
        documents.extend(docs)

    # Create vector store
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    # Create the index
    index = GPTVectorStoreIndex.from_documents(
        documents,
        embedding=model.encode,
        text_splitter=GPTTextSplitter.from_chunk_size(700, chunk_overlap=50),
        vector_store=vector_store
    )

    # Save the index to disk
    index.storage_context.persist(f"{collection_name}_index.json")
    print("Vector DB Successfully Created!")
