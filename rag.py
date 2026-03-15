#RAG - sending the k=1 likely results in context to a model
import faiss
import numpy as np
import ollama

# Your existing retrieval code
texts = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
]

# Create embeddings and index (your existing code)
vectors = []
for t in texts:
    emb = ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"]
    vectors.append(emb)

vectors = np.array(vectors).astype("float32")
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(vectors)

# Query
query = "How do I quickly find an element in a sorted list?"
query_vector = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

# Retrieve relevant context
D, I = index.search(np.array([query_vector]).astype("float32"), k=1)
relevant_code = texts[I[0][0]]

# NOW GENERATE THE RESPONSE using the retrieved context
prompt = f"""Based on this code snippet:
{relevant_code}

Answer this question: {query}

Provide a helpful explanation with the code."""

response = ollama.generate(model="qwen2.5:3b", prompt=prompt)
print("Response:", response['response'])