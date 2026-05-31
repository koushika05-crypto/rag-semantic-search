from sentence_transformers import SentenceTransformer

# Load the AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your test sentences
sentences = [
    "How do neural networks learn?",
    "Python is great for data science",
    "Machine learning uses data to make predictions",
    "The cat sat on the mat",
    "Deep learning is a subset of machine learning"
]

# Convert sentences into vectors (embeddings)
embeddings = model.encode(sentences)

print(f"✅ Success! Shape: {embeddings.shape}")
print(f"Each sentence = {embeddings.shape[1]} numbers")
print(f"\nFirst embedding preview:\n{embeddings[0][:5]}...")