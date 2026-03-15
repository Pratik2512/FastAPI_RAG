#Creating embeddings 
import faiss
import numpy as np
import ollama

#array of snippets or chunks
texts = [
    "def quicksort(arr):",
    "def binary_search(arr, target):"
]

vectors = []

#converting the chunks into embeddings 
for t in texts:
    emb = ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"]
    vectors.append(emb)

#converting the array of embeddings into float type - since faiss deals with this type
vectors = np.array(vectors).astype("float32")

#creating the index for similarity search
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(vectors)


#creating embedding for the query 
query = ollama.embeddings(
    model="nomic-embed-text",
    prompt="fast search in sorted array"
)["embedding"]

#performing the similarity search
D, I = index.search(np.array([query]).astype("float32"), k=1)

print("Most similar:", texts[I[0][0]])