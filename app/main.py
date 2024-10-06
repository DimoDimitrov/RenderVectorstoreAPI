from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import os
from typing import List

app = FastAPI()

PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "/data/vectorstore")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

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
        print(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail="Failed to add document")

@app.get("/query")
async def query(query_text: str, collection_name: str, n_results: int = 5):
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        return results
    except Exception as e:
        print(f"Error querying: {e}")
        raise HTTPException(status_code=500, detail="Failed to query")

@app.delete("/delete_documents")
async def delete_documents(document_ids: List[str], collection_name: str):
    try:
        collection = chroma_client.get_collection(collection_name)
        collection.delete(ids=document_ids)
        return {"message": f"{len(document_ids)} documents deleted successfully"}
    except Exception as e:
        print(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete documents")

@app.put("/update_documents")
async def update_documents(documents: List[Document], collection_name: str):
    try:
        collection = chroma_client.get_collection(collection_name)
        for doc in documents:
            collection.update(
                ids=[doc.id],
                documents=[doc.content],
                metadatas=[doc.metadata]
            )
        return {"message": f"{len(documents)} documents updated successfully"}
    except Exception as e:
        print(f"Error updating documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to update documents")

# if __name__ == "__main__":
#     # Used for running the server locally. Usefull for debugging.
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)