from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv
import fitz

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')
documents = []
index = faiss.IndexFlatL2(384)
full_document_text = ""

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    if index.ntotal == 0:
        return {"answer": "Please upload a document first!", "sources": []}

    # Search top 5 most relevant chunks
    query_vec = model.encode([q.question])
    distances, indices_result = index.search(np.array(query_vec), k=5)
    chunks = [documents[i] for i in indices_result[0] if i < len(documents)]
    context = "\n".join(chunks)

    # Use first 2000 chars of full document for extra context
    extra_context = full_document_text[:2000] if full_document_text else ""

    prompt = f"""You are a helpful document assistant. Answer the question using ONLY the document content below.

RULES:
- Answer directly and specifically
- For resumes: find name, skills, education, CGPA, experience from the text
- Never say "I couldn't find" if the info exists anywhere in the context
- Be specific — include exact names, numbers, dates, percentages

Full document (first section):
{extra_context}

Most relevant chunks:
{context}

Question: {q.question}

Direct answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks[:3]
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global full_document_text, documents, index

    try:
        contents = await file.read()
        pdf = fitz.open(stream=contents, filetype="pdf")

        chunks = []
        full_text = ""

        for page in pdf:
            text = page.get_text()
            full_text += text + "\n"

            lines = text.split('\n')
            current_chunk = ""

            for line in lines:
                line = line.strip()
                if not line:
                    if len(current_chunk) > 50:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    continue
                current_chunk += " " + line
                if len(current_chunk) > 200:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

            if len(current_chunk) > 30:
                chunks.append(current_chunk.strip())

        # Store full text globally
        full_document_text = full_text

        # Reset index and documents for new upload
        index = faiss.IndexFlatL2(384)
        documents = []

        if chunks:
            embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)
            index.add(np.array(embeddings))
            documents.extend(chunks)

        # Generate smart summary using Groq
        summary_text = full_text[:3000]
        summary_prompt = f"""Read this document carefully and write a clear 2-3 sentence summary.
State: what type of document it is, who it belongs to, and the key highlights.
Be specific — include the person's name, institution, skills if it is a resume.

Document text:
{summary_text}

Summary:"""

        summary_response = groq_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200,
            temperature=0.1,
        )

        return {
            "message": f"Uploaded! Added {len(chunks)} chunks to the knowledge base.",
            "chunks": len(chunks),
            "summary": summary_response.choices[0].message.content
        }

    except Exception as e:
        return {
            "message": f"Upload failed: {str(e)}",
            "chunks": 0,
            "summary": ""
        }

@app.get("/")
def root():
    return {"status": "RAG server is running!"}