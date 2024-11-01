from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.errors import InvalidCollectionException
import os
from typing import List
import logging
from functools import lru_cache

from .vectorization_check import check_agent, delete_agent

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HNSW_SETTINGS = {
    "space": "cosine",
    "M": 128,
    "ef_construction": 200,
    "ef": 100
}

app = FastAPI()

app.add_api_route("/check_agent", check_agent, methods=["POST"])
app.add_api_route("/delete_agent/{agent_id}", delete_agent, methods=["DELETE"])

PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "/data/vectorstore")

@lru_cache(maxsize=None)
def get_chroma_client(collection_name: str):
    try:
        unique_persist_dir = os.path.join(PERSIST_DIRECTORY, collection_name)
        os.makedirs(unique_persist_dir, exist_ok=True)
        logger.info(f"Creating ChromaDB client for collection: {collection_name}")
        logger.info(f"Persist directory: {unique_persist_dir}")
        client = chromadb.PersistentClient(path=unique_persist_dir)
        # Test the connection
        client.heartbeat()
        logger.info(f"ChromaDB client created and tested successfully for collection: {collection_name}")
        return client
    except Exception as e:
        logger.error(f"Failed to create ChromaDB client: {str(e)}", exc_info=True)
        raise

@lru_cache(maxsize=None)
def get_or_create_collection(collection_name: str):
    try:
        client = get_chroma_client(collection_name)
        
        try:
            # Try to get existing collection directly by name
            collection = client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
            return collection
            
        except InvalidCollectionException:
            # Create new collection if it doesn't exist
            collection = client.create_collection(
                name=collection_name,
                metadata=HNSW_SETTINGS
            )
            logger.info(f"Created new collection: {collection_name}")
            return collection
            
        except Exception as e:
            # If there's any other error, try recreating the collection
            logger.warning(f"Error accessing collection {collection_name}, attempting to recreate: {e}")
            try:
                client.delete_collection(collection_name)
            except:
                pass
                
            collection = client.create_collection(
                name=collection_name,
                metadata=HNSW_SETTINGS
            )
            logger.info(f"Recreated collection: {collection_name}")
            return collection
            
    except Exception as e:
        logger.error(f"Error in get_or_create_collection: {e}", exc_info=True)
        raise

class Document(BaseModel):
    id: str
    content: str
    metadata: dict

@app.post("/create_persist_directory")
def create_persist_directory(collection_name: str) -> dict:
    try:
        unique_persist_dir = os.path.join(PERSIST_DIRECTORY, collection_name)
        if not os.path.exists(unique_persist_dir):
            os.makedirs(unique_persist_dir, exist_ok=True)
        logger.info(f"Created persist directory for collection: {collection_name}")
        return {"message": f"Persist directory created for collection: {collection_name}", "path": unique_persist_dir}
    except Exception as e:
        logger.error(f"Error creating persist directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create persist directory: {str(e)}")

@app.post("/add_document")
async def add_document(document: Document, collection_name: str):
    try:
        collection = get_or_create_collection(collection_name)
        existing_docs = collection.get(ids=[document.id])
        if existing_docs['ids']:
            collection.update(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.metadata]
            )
            logger.info(f"Document updated: {document.id}")
            return {"message": "Document updated successfully"}
        else:
            collection.add(
                documents=[document.content],
                metadatas=[document.metadata],
                ids=[document.id]
            )
            logger.info(f"Document added: {document.id}")
            return {"message": "Document added successfully"}
    except Exception as e:
        logger.error(f"Error adding/updating document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add/update document: {str(e)}")

@app.get("/query")
async def query(query_text: str, collection_name: str, n_results: int = 5):
    try:
        collection = get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        logger.info(f"Query successful: {query_text}")
        return results
    except Exception as e:
        logger.error(f"Error querying: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to query: {str(e)}")

@app.delete("/delete_documents")
async def delete_documents(document_ids: List[str], collection_name: str):
    try:
        client = get_chroma_client(collection_name)
        try:
            collection = client.get_collection(collection_name)
        except InvalidCollectionException:
            logger.warning(f"Collection {collection_name} does not exist")
            return {"message": "Collection does not exist, no documents deleted"}

        collection.delete(ids=document_ids)
        return {"message": f"{len(document_ids)} documents deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete documents")

@app.put("/update_documents")
async def update_documents(documents: List[Document], collection_name: str):
    try:
        create_persist_directory(collection_name)
        client = get_chroma_client(collection_name)
        try:
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata=HNSW_SETTINGS
            )
        except InvalidCollectionException:
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "space": "cosine",
                    "M": 128,
                    "ef_construction": 200,
                    "ef": 100
                }
            )
            logger.info(f"Created new collection: {collection_name}")

        for doc in documents:
            collection.update(
                ids=[doc.id],
                documents=[doc.content],
                metadatas=[doc.metadata]
            )
        return {"message": f"{len(documents)} documents updated successfully"}
    except Exception as e:
        logger.error(f"Error updating documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update documents")

# if __name__ == "__main__":
#     # Used for running the server locally. Usefull for debugging.
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)