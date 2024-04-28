import os
from pathlib import Path
from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from llama_index.vector_stores import QdrantVectorStore
from llama_index import StorageContext
from qdrant_client import QdrantClient

# Load PDF files
dir_path = Path("path/to/pdf/files")
pdf_files = list(dir_path.glob("*.pdf"))
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()
    documents.extend(docs)

# Create vector store
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
client = QdrantClient(":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="pdf_documents")

# Create the index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    text_splitter=GPTTextSplitter.from_chunk_size(700, chunk_overlap=50),
    embedding=model.encode
)

# Save the index to disk
index.storage_context.persist("pdf_index.json")

print("Vector DB Successfully Created!")
