from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.errors import InvalidCollectionException
import os
from typing import List, Union
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
async def add_document(document: Union[Document, List[Document]], collection_name: str):
    try:
        collection = get_or_create_collection(collection_name)
        
        # Convert single document to list if needed
        documents = [document] if isinstance(document, Document) else document
        
        # Get all document IDs first
        all_doc_ids = [str(doc.id) for doc in documents]
        
        # Check existence for all documents at once
        existing = collection.get(
            ids=all_doc_ids,
            include=["metadatas"]
        )
        
        existing_ids = set(existing['ids'])
        
        # Split into documents to update and add
        to_update_ids = []
        to_update_contents = []
        to_update_metadata = []
        
        to_add_ids = []
        to_add_contents = []
        to_add_metadata = []
        
        # Sort documents into appropriate lists
        for doc in documents:
            doc_id = str(doc.id)
            if doc_id in existing_ids:
                to_update_ids.append(doc_id)
                to_update_contents.append(doc.content)
                to_update_metadata.append(doc.metadata)
            else:
                to_add_ids.append(doc_id)
                to_add_contents.append(doc.content)
                to_add_metadata.append(doc.metadata)
        
        # Perform updates in batches if needed
        if to_update_ids:
            batch_size = 20
            for i in range(0, len(to_update_ids), batch_size):
                batch_end = i + batch_size
                collection.update(
                    ids=to_update_ids[i:batch_end],
                    documents=to_update_contents[i:batch_end],
                    metadatas=to_update_metadata[i:batch_end]
                )
            logger.info(f"Updated {len(to_update_ids)} documents in batch")
        
        # Perform adds in batches if needed
        if to_add_ids:
            batch_size = 20
            for i in range(0, len(to_add_ids), batch_size):
                batch_end = i + batch_size
                collection.add(
                    ids=to_add_ids[i:batch_end],
                    documents=to_add_contents[i:batch_end],
                    metadatas=to_add_metadata[i:batch_end]
                )
            logger.info(f"Added {len(to_add_ids)} documents in batch")
        
        return {
            "message": f"Updated {len(to_update_ids)} documents, Added {len(to_add_ids)} documents",
            "updated_ids": to_update_ids,
            "added_ids": to_add_ids
        }
            
    except Exception as e:
        logger.error(f"Error adding/updating documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add/update documents: {str(e)}")

@app.get("/query")
async def query(
    query_text: str, 
    collection_name: str, 
    n_results: int = 5,
    offset: int = 0
):
    try:
        collection = get_or_create_collection(collection_name)
        total_count = collection.count()
        
        if query_text.strip():
            # If there's a query text, use query method for similarity search
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
        else:
            # If no query text, use get method for paginated document retrieval
            results = collection.get(
                limit=n_results,
                offset=offset,
                include=["metadatas", "documents"]
            )
        
        response = {
            **results,
            "pagination": {
                "offset": offset,
                "limit": n_results,
                "total": total_count
            }
        }
        
        logger.info(f"Query successful: {query_text} (offset: {offset}, limit: {n_results})")
        return response
    except Exception as e:
        logger.error(f"Error querying: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to query: {str(e)}")

@app.delete("/delete_documents")
async def delete_documents(document_ids: List[str], collection_name: str):
    try:
        client = get_chroma_client(collection_name)
        try:
            collection = client.get_or_create_collection(collection_name)
        except InvalidCollectionException:
            logger.warning(f"Collection {collection_name} does not exist")
            return {"message": "Collection does not exist, no documents deleted"}

        existing_docs = collection.get(ids=document_ids)
        existing_ids = existing_docs['ids']
        
        if not existing_ids:
            logger.warning(f"No documents found with the provided IDs in collection {collection_name}")
            return {"message": "No documents found to delete"}

        # Only delete documents that exist
        collection.delete(ids=existing_ids)
        logger.info(f"Successfully deleted {len(existing_ids)} documents")
        
        # Report any IDs that weren't found
        not_found_ids = set(document_ids) - set(existing_ids)
        if not_found_ids:
            logger.warning(f"Documents not found for deletion: {not_found_ids}")
            
        return {
            "message": f"{len(existing_ids)} documents deleted successfully",
            "deleted_ids": existing_ids,
            "not_found_ids": list(not_found_ids)
        }
    except Exception as e:
        logger.error(f"Error deleting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete documents")

@app.put("/update_documents")
async def update_documents(documents: List[Document], collection_name: str):
    try:
        collection = get_or_create_collection(collection_name)
        
        # Get all document IDs first
        all_doc_ids = [str(doc.id) for doc in documents]
        
        # Check existence for all documents at once
        existing = collection.get(
            ids=all_doc_ids,
            include=["metadatas"]
        )
        
        existing_ids = set(existing['ids'])
        
        # Prepare update lists
        to_update_ids = []
        to_update_texts = []
        to_update_metadata = []
        skipped_ids = []
        
        # Sort documents into update and skip lists
        for doc in documents:
            doc_id = str(doc.id)
            if doc_id in existing_ids:
                to_update_ids.append(doc_id)
                to_update_texts.append(doc.content)
                to_update_metadata.append(doc.metadata)
            else:
                skipped_ids.append(doc_id)
                logger.warning(f"Document {doc_id} not found - skipping update")
        
        # Perform updates in batches of 20
        batch_size = 20
        for i in range(0, len(to_update_ids), batch_size):
            batch_end = i + batch_size
            collection.update(
                ids=to_update_ids[i:batch_end],
                documents=to_update_texts[i:batch_end],
                metadatas=to_update_metadata[i:batch_end]
            )
            logger.info(f"Updated the following documents: {to_update_ids[i:batch_end]} ")
        
        return {
            "message": f"Updated {len(to_update_ids)} documents",
            "updated_ids": to_update_ids,
            "skipped_ids": skipped_ids
        }
    except Exception as e:
        logger.error(f"Error updating documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update documents")

@app.get("/collection_info")
async def get_collection_info(collection_name: str):
    try:
        collection = get_or_create_collection(collection_name)
        metadata = collection.metadata
        return {
            "metadata": metadata,
            "count": collection.count()
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recreate_collection")
async def recreate_collection(collection_name: str):
    try:
        # Clear the caches before recreating
        get_chroma_client.cache_clear()
        get_or_create_collection.cache_clear()
        
        client = get_chroma_client(collection_name)
        
        # Get all existing documents
        old_collection = client.get_collection(name=collection_name)
        existing_docs = old_collection.get()
        
        # Delete the old collection
        client.delete_collection(collection_name)
        
        # Create new collection with correct HNSW settings
        new_collection = client.create_collection(
            name=collection_name,
            metadata=HNSW_SETTINGS
        )
        
        # Re-add all documents if there were any
        if existing_docs and existing_docs['ids']:
            new_collection.add(
                ids=existing_docs['ids'],
                documents=existing_docs['documents'],
                metadatas=existing_docs['metadatas']
            )
        
        return {
            "message": "Collection recreated successfully",
            "documents_migrated": len(existing_docs['ids']) if existing_docs else 0
        }
    except Exception as e:
        logger.error(f"Error recreating collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     # Used for running the server locally. Usefull for debugging.
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)