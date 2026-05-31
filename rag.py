from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

# Your knowledge base
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
    "Semantic search finds meaning, not just exact keyword matches"
]

# Build FAISS index
print("Building index...")
embeddings = model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def ask(question):
    print(f"\nQuestion: {question}")

    # STEP 1: RETRIEVE
    query_vec = model.encode([question])
    distances, indices = index.search(np.array(query_vec), k=3)
    relevant_chunks = [documents[i] for i in indices[0]]

    print("\nRetrieved context:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"  {i+1}. {chunk}")

    # STEP 2: AUGMENT
    context = "\n".join(relevant_chunks)
    prompt = f"""Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    # STEP 3: GENERATE using Ollama (free!)
    print("\nAsking Llama...")
    response = ollama.chat(
        model='llama3.2',
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response['message']['content']
    print(f"\nAnswer: {answer}")
    return answer

# Test it!
ask("How do machines learn from data?")
ask("What is semantic search?")