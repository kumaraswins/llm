from dotenv import load_dotenv
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
load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
model_type = os.environ.get('MODEL_TYPE')
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
    # collection_name = st.text_input("Collection Name") not working for some reason
    if st.button("Embed"):
        embed_documents(files, "collection_name")
    
    # Query section
    st.header("Document Retrieval")
    collection_names = get_collection_names()
    selected_collection = st.selectbox("Select a document", collection_names)
    if selected_collection:
        query = st.text_input("Query")
        if st.button("Retrieve"):
            retrieve_documents(query, selected_collection)

def embed_documents(files:List[st.runtime.uploaded_file_manager.UploadedFile], collection_name:str):
    # endpoint = f"{API_BASE_URL}/embed"
    # files_data = [("files", file) for file in files]
    # data = {"collection_name": collection_name}

    # response = requests.post(endpoint, files=files_data, data=data)
    # if response.status_code == 200:
    #     st.success("Documents embedded successfully!")
    # else:
    #     st.error("Document embedding failed.")
    #     st.write(response.text)
    saved_files = []
    for file in files:
        file_path = os.path.join(source_directory, file.filename)
        saved_files.append(file_path)
        
        with open(file_path, "wb") as f:
            f.write(file.read())

    out = ingest.main(collection_name)
    st.write(out)


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