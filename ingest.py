from langchain.document_loaders import PyPDFLoader, PDFMinerLoader, DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

def main():
    for root,dir,files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(path=os.join(root,file))

    documents = loader.load()
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=500)
    texts = textsplitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="")

    db = Chroma.from_documents(texts, embeddings, persist_directory="db", client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None

if __name__=="__main__":
    main()
