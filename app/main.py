from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import os
import asyncio

app = FastAPI()

# Initialize ChromaDB client
persist_directory = os.environ.get("PERSIST_DIRECTORY", "/data/vectorstore")
chroma_client = chromadb.PersistentClient(path=persist_directory)

class Document(BaseModel):
    id: str
    content: str
    metadata: dict

@app.post("/add_document")
async def add_document(document: Document, collection_name: str):
    try:
        collection = chroma_client.get_or_create_collection(collection_name)
        collection.add(
        documents=[document.content],
        metadatas=[document.metadata],
        ids=[document.id]
        )
        return {"message": "Document added successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to add document")

@app.get("/query")
async def query(query_text: str, collection_name: str, n_results: int = 5):
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
        query_texts=[query_text],
        n_results=n_results
        )
        return results
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to query")


async def test_query():
    results = await query("What is the capital of France?", "test", 5)
    print(results)

# asyncio.run(test_query())
# Add more endpoints as needed (update, delete, etc.)