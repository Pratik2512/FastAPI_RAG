# Filename: app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import ollama

app = FastAPI(title="Embeddings Vector API")

# Define request schema
class TextData(BaseModel):
    text: str
    id: str  # unique identifier for the text

# Initialize FAISS index and a mapping from ID -> index
vectors = []
ids = []
index = None  # Will be initialized after first embedding

@app.post("/add_text")
def add_text(data: TextData):
    global index, vectors, ids
    
    # Generate embedding using Ollama
    try:
        emb_response = ollama.embeddings(model="nomic-embed-text", prompt=data.text)
        emb = np.array(emb_response["embedding"]).astype("float32")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    
    # Initialize index if empty
    if index is None:
        index = faiss.IndexFlatL2(len(emb))
    
    # Add vector and ID
    index.add(np.array([emb]))
    vectors.append(emb)
    ids.append(data.id)
    
    return {"status": "success", "id": data.id, "vector_dim": len(emb)}

@app.post("/query")
def query_text(query: str, k: int = 1):
    global index, vectors, ids
    if index is None or len(vectors) == 0:
        raise HTTPException(status_code=400, detail="No vectors in the database yet.")
    
    # Generate embedding for query
    try:
        query_emb = np.array(ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]).astype("float32")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(e)}")
    
    # Search top-k similar
    D, I = index.search(np.array([query_emb]), k)
    results = [{"id": ids[i], "distance": float(D[0][idx])} for idx, i in enumerate(I[0])]
    
    return {"query": query, "results": results}

# Optional endpoint to list all IDs
@app.get("/list_ids")
def list_ids():
    return {"ids": ids}