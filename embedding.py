import ollama

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt="def binary_search(arr, target):"
)

vector = response["embedding"]

print(len(vector))
print(vector[:10])