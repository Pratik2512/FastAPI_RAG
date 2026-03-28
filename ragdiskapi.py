# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import ollama
import os
import json

app = FastAPI(title="Persistent Vector DB API")

INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.json"

# Request schemas
class TextData(BaseModel):
    id: str
    xsjs: str
    nodejs: str
    description: str

class QueryData(BaseModel):
    query: str
    k: int = 1

class RAGQuery(BaseModel):
    query: str
    k: int = 3
    model: str = "qwen2.5:3b"

class DeleteData(BaseModel):
    id: str

# Global storage
index = None
metadata = []


# -----------------------------
# LOAD EXISTING DATABASE
# -----------------------------
def load_db():
    global index, metadata

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print("FAISS index loaded")

    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
        print("Metadata loaded")


# -----------------------------
# SAVE DATABASE
# -----------------------------
def save_db():
    global index, metadata

    if index is not None:
        faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w") as f:
        json.dump(metadata, f)


@app.on_event("startup")
def startup():
    load_db()


# -----------------------------
# ADD TEXT ENDPOINT
# -----------------------------
@app.post("/add")
def add_text(data: TextData):

    global index, metadata

    # 🔹 Combine fields into one embedding text
    combined_text = f"""
    XSJS Code:
    {data.xsjs}

    Node.js Code:
    {data.nodejs}

    Description:
    {data.description}
    """

    # 🔹 Create embedding
    try:
        emb = ollama.embeddings(
            model="nomic-embed-text",
            prompt=combined_text
        )["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    vector = np.array(emb).astype("float32")

    # 🔹 Initialize FAISS index if needed
    if index is None:
        index = faiss.IndexFlatL2(len(vector))

    index.add(np.array([vector]))

    # 🔹 Store structured metadata (important for RAG retrieval)
    metadata.append({
        "id": data.id,
        "xsjs": data.xsjs,
        "nodejs": data.nodejs,
        "description": data.description
    })

    save_db()

    return {
        "status": "stored",
        "vector_dimension": len(vector)
    }


# -----------------------------
# SEARCH ENDPOINT
# -----------------------------
@app.post("/search")
def search(data: QueryData):

    global index, metadata

    if index is None:
        raise HTTPException(status_code=400, detail="Database empty")

    emb = ollama.embeddings(
        model="nomic-embed-text",
        prompt=data.query
    )["embedding"]

    query_vector = np.array([emb]).astype("float32")

    D, I = index.search(query_vector, data.k)

    results = []

    for pos, idx in enumerate(I[0]):
        results.append({
            "id": metadata[idx]["id"],
            "text": metadata[idx]["text"],
            "distance": float(D[0][pos])
        })

    return {
        "query": data.query,
        "results": results
    }


# -----------------------------
# LIST STORED ITEMS
# -----------------------------
@app.get("/list")
def list_items():
    return metadata


#------------------------------
#RAG ENDPOINT
#------------------------------

@app.post("/ask")
def ask_llm(data: RAGQuery):

    global index, metadata

    if index is None or len(metadata) == 0:
        raise HTTPException(status_code=400, detail="Vector database empty")

    # 🔹 Create embedding for XSJS query
    emb = ollama.embeddings(
        model="nomic-embed-text",
        prompt=data.query
    )["embedding"]

    query_vector = np.array([emb]).astype("float32")

    # 🔹 Search FAISS
    D, I = index.search(query_vector, data.k)

    contexts = []
    for idx in I[0]:
        # 🔹 Skip invalid indices
        if idx == -1 or idx >= len(metadata):
            continue

        item = metadata[idx]

        # 🔹 Format structured context for better grounding
        formatted = f"""
XSJS:
{item['xsjs']}

Node.js:
{item['nodejs']}

Description:
{item['description']}
"""
        contexts.append(formatted)

    # 🔹 Combine context
    context_text = "\n\n---\n\n".join(contexts)

    # 🔹 Stronger prompt for code conversion
    prompt = f"""
You are an expert SAP XSJS to Node.js (CAP) migration assistant.

Your task is to convert XSJS code into modern async Node.js code using cds.run.

Guidelines:
- Use async/await
- Replace prepareStatement / prepareCall with cds.run
- Preserve logic and parameters
- Return clean, production-ready code
- Do NOT explain unless asked
- Output only Node.js code

Reference Examples:
{context_text}

XSJS Input:
{data.query}

Convert the above XSJS code into Node.js async function.
"""

    # 🔹 Generate response
    response = ollama.generate(
        model=data.model,
        prompt=prompt
    )

    return {
        "query": data.query,
        "matched_examples": contexts,
        "answer": response["response"]
    }


@app.delete("/delete")
def delete_item(data: DeleteData):

    global index, metadata

    if len(metadata) == 0:
        raise HTTPException(status_code=400, detail="Database empty")

    # Find index of item to delete
    delete_idx = None
    for i, item in enumerate(metadata):
        if item["id"] == data.id:
            delete_idx = i
            break

    if delete_idx is None:
        raise HTTPException(status_code=404, detail="ID not found")

    # Remove from metadata
    metadata.pop(delete_idx)

    # Rebuild FAISS index
    index = None

    if len(metadata) > 0:
        vectors = []

        for item in metadata:
            emb = ollama.embeddings(
                model="nomic-embed-text",
                prompt=item["text"]
            )["embedding"]

            vectors.append(emb)

        vectors = np.array(vectors).astype("float32")

        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(vectors)

    # Save updated DB
    save_db()

    return {
        "status": "deleted",
        "id": data.id,
        "remaining_items": len(metadata)
    }