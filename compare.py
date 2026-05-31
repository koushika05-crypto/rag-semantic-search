from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Same knowledge base
documents = [
    "Neural networks learn by adjusting weights using backpropagation",
    "Python is the most popular language for data science and AI",
    "Machine learning uses data and algorithms to make predictions",
    "FAISS is a library for fast similarity search in large datasets",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing helps computers understand human language",
    "Transformers are a type of neural network used in modern NLP",
    "Vector embeddings represent text as numbers in high-dimensional space",
    "RAG stands for Retrieval Augmented Generation",
    "Semantic search finds meaning not just exact keyword matches"
]

# Test queries
queries = [
    "how do machines learn from data?",
    "what is meaning based search?",
    "how are words converted to numbers?",
    "what technology finds similar vectors fast?",
    "how does AI understand human language?",
]

# Expected best answers for each query
expected = [
    "Machine learning uses data and algorithms to make predictions",
    "Semantic search finds meaning not just exact keyword matches",
    "Vector embeddings represent text as numbers in high-dimensional space",
    "FAISS is a library for fast similarity search in large datasets",
    "Natural language processing helps computers understand human language",
]

print("Setting up models...")

# SEMANTIC SEARCH setup
sem_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = sem_model.encode(documents)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# KEYWORD SEARCH setup
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

results = []

print("\nRunning comparison...\n")
print("=" * 60)

for i, query in enumerate(queries):
    # Semantic search
    query_vec = sem_model.encode([query])
    _, sem_indices = index.search(np.array(query_vec), k=1)
    sem_result = documents[sem_indices[0][0]]
    sem_correct = sem_result == expected[i]

    # Keyword search
    query_tfidf = tfidf.transform([query])
    scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    kw_index = np.argmax(scores)
    kw_result = documents[kw_index]
    kw_correct = kw_result == expected[i]

    results.append({
        "query": query,
        "semantic_correct": sem_correct,
        "keyword_correct": kw_correct,
        "semantic_result": sem_result,
        "keyword_result": kw_result,
        "expected": expected[i]
    })

    print(f"Query: {query}")
    print(f"  Semantic : {'✅ CORRECT' if sem_correct else '❌ WRONG'} → {sem_result[:50]}...")
    print(f"  Keyword  : {'✅ CORRECT' if kw_correct else '❌ WRONG'} → {kw_result[:50]}...")
    print()

# Score summary
sem_score = sum(r["semantic_correct"] for r in results)
kw_score = sum(r["keyword_correct"] for r in results)

print("=" * 60)
print(f"FINAL SCORES (out of {len(queries)})")
print(f"  Semantic Search : {sem_score}/{len(queries)} correct")
print(f"  Keyword Search  : {kw_score}/{len(queries)} correct")
print(f"  Winner          : {'Semantic Search 🏆' if sem_score > kw_score else 'Keyword Search'}")

# Save results for the chart
with open("comparison_results.json", "w") as f:
    json.dump({
        "results": results,
        "semantic_score": sem_score,
        "keyword_score": kw_score,
        "total": len(queries)
    }, f, indent=2)

print("\nResults saved to comparison_results.json")