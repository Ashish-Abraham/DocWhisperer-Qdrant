import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List

model_name = "sentence-transformers/all-MiniLM-L6-v2"
collection_name = "ASM"
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

qdrant_client = QdrantClient(host='localhost', port=6333)
collections = qdrant_client.get_collections()
print(collections)

# load the reader model into a question-answering pipeline
reader = pipeline("question-answering", model=model_name, tokenizer=model_name)

def get_context(query: str, top_k: int) -> List[str]:
    """
    Get the relevant context from the database for a given query

    Args:
        query (str): What do we want to know?
        top_k (int): Top K results to return

    Returns:
        context (List[str]):
    """
    try:
        encoded_query = embedding_model.encode(query).tolist()  

        result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=encoded_query,
            limit=top_k,
        )  # search qdrant collection for context passage with the answer

        context = [
            [x.payload["text"]] for x in result
        ]  
        return context

    except Exception as e:
        print({e})



def get_response(query: str, context: List[str]):
    """
    Extract the answer from the context for a given query

    Args:
        query (str): _description_
        context (list[str]): _description_
    """
    results = []
    for c in context:
        # feed the reader the query and contexts to extract answers
        answer = reader(question=query, context=c[0])
        results.append(answer)

    # sort the result based on the score from reader model
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    for i in range(len(results)):
        print(f"{i+1}", end=" ")
        print(
            "Answer: ",
            results[i]["answer"],
            "\n  score: ",
            results[i]["score"],
        )


query = "QUERY HERE"
context = get_context(query, top_k=1)
print("Context: {}\n".format(context))

get_response(query, context)
client.delete_collection(collection_name=collection_name)

