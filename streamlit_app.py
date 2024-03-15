import os
import streamlit as st
import requests
from typing import List
import json
import socket
from urllib3.connection import HTTPConnection
import ingest
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from constants import CHROMA_SETTINGS
#load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY', "db")
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME","all-MiniLM-L6-v2")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
model_type = os.environ.get('MODEL_TYPE',"mistral")

llm = Ollama(model=model_type, callbacks=[])
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


def list_of_collections():
    client = chromadb.Client(CHROMA_SETTINGS)
    return (client.list_collections())
    
def main():
    st.title("PrivateGPT App: Document Embedding and Retrieval")
    # Document upload section
    st.header("Document Upload")
    files = st.file_uploader("Upload document", accept_multiple_files=True)
    collection_name = st.text_input("Collection Name") 
    if collection_name and st.button("Embed"):
        embed_documents(files, collection_name)
    
    # Query section
    st.header("Document Retrieval")
    collection_names = get_collection_names()
    selected_collection = st.selectbox("Select a document", collection_names)
    if selected_collection:
        query = st.text_input("Query")
        if st.button("Retrieve"):
            retrieve_documents(query, selected_collection)

def embed_documents(files:List[st.runtime.uploaded_file_manager.UploadedFile], collection_name:str):
    
    saved_files = []
    # print(files_data)
    for uploaded_file in files:
        file_path = os.path.join(source_directory, uploaded_file.name)
        saved_files.append(file_path)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    out = ingest.main(collection_name)
    st.write(out)
    [os.remove(os.path.join(source_directory, file.name)) or os.path.join(source_directory, file.name) for file in files]


def get_collection_names():

    collections =  os.listdir(persist_directory)
    collections.remove(".DS_Store")
    return collections



def retrieve_documents(query: str, collection_name: str):    
    knowledge = f"{persist_directory}/{collection_name}"
    if(os.path.isdir(knowledge)):
        db = Chroma(persist_directory=knowledge, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        st.subheader("Results")
        st.write(answer)
        st.subheader("Documents")
        for doc in docs:
            st.write(doc)
    else:
        st.subheader("Results")
        st.text("No data found")
        st.subheader("Documents")
        st.text("No related docs")

if __name__ == "__main__":
    main()