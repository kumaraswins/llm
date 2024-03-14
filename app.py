import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from fastapi import FastAPI, UploadFile
from typing import List, Optional
from pydantic import BaseModel
from langchain.llms import Ollama
from constants import CHROMA_SETTINGS
import ingest

app = FastAPI()
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
llm = Ollama(model=model_type, callbacks=[StreamingStdOutCallbackHandler])

# Starting the app with embedding and llm download
@app.on_event("startup")
async def startup_event():
    print("Starting ::>")

class Retrieve(BaseModel):
    query: str
    collection_name: str 

class Embed(BaseModel):
    files: List[UploadFile]
    collection_name: str 

# Example route
@app.get("/")
async def root():
    return {"message": "Hello, the APIs are now ready for your embeds and queries!"}

@app.post("/embed")
async def embed(files: List[UploadFile], collection_name: Optional[str] = None):
    saved_files = []
    for file in files:
        file_path = os.path.join(source_directory, file.filename)
        saved_files.append(file_path)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
    ingest.main(collection_name)
    return {"message": "Files embedded successfully", "saved_files": saved_files}


@app.post("/retrieve")
async def query(data:Retrieve):
    knowledge = f"{persist_directory}/{data.collection_name}"
    if(os.path.isdir(knowledge)):
        db = Chroma(persist_directory=knowledge, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        res = qa(data.query)
        answer, docs = res['result'], res['source_documents']
        return {"results": answer, "docs":docs}
    else:
        return {"results": "No collections found", "docs":[]}

@app.get("/collections")
async def get_collections():
    collection = os.listdir(persist_directory)
    collection.remove(".DS_Store")
    return {"collections": collection}
