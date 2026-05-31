from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Your knowledge base (later this will be your PDFs/articles)
documents = [
    "Neural networks learn by adjusting weights using backpropagation",
    "Python is the most popular language for data science and AI",
    "Machine learning uses data and algorithms to make predictions",
    "FAISS is a library for fast similarity search in large datasets",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing helps computers understand human language",
    "Transformers are a type of neural network used in modern NLP models",
    "Vector embeddings represent text as numbers in high-dimensional space",
    "RAG stands for Retrieval Augmented Generation",
    "Semantic search finds meaning, not just exact keyword matches"
]

# Step 1: Load model and embed all documents
print("🔄 Embedding documents...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Step 2: Build FAISS index
print("🔄 Building FAISS index...")
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print(f"✅ Index built with {index.ntotal} documents\n")

# Step 3: Search!
def search(query, top_k=3):
    print(f"🔍 Query: '{query}'")
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)

    print("📄 Top results:")
    for i, idx in enumerate(indices[0]):
        print(f"  {i+1}. {documents[idx]}")
        print(f"     Score: {distances[0][i]:.4f} (lower = more similar)\n")

# Try these searches
search("how do machines learn from data?")
search("what is semantic search?")