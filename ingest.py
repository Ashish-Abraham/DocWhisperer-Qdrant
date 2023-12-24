import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')



loader = PyPDFLoader("docs/ASM-standards.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

qdrant_client = QdrantClient(host='localhost', port=6333)
my_collection = "ASM"
qdrant_client.recreate_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)



# Generate embeddings for each text chunk
embeddings_list = []
points_list = []
i=0
for text in texts:
    embedding = model.encode(text.page_content)
    point_dict = {
        "id": i+1,
        "vector": embedding,
        "payload": {"text": text.page_content},
    }
    points_list.append(point_dict)
    i+=1
    embeddings_list.append(embedding)



print("Embeddings Successful")    





qdrant_client.upsert(collection_name=my_collection, points=points_list)




print("Vector DB Successfully Created!")
