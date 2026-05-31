<div align="center">

```
██████╗  ██████╗  ██████╗███╗   ███╗██╗███╗   ██╗██████╗ 
██╔══██╗██╔═══██╗██╔════╝████╗ ████║██║████╗  ██║██╔══██╗
██║  ██║██║   ██║██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
██║  ██║██║   ██║██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
██████╔╝╚██████╔╝╚██████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚═════╝  ╚═════╝  ╚═════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ 
```

### 🧠 AI-Powered Document Assistant — Chat with your PDFs

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Meta_AI-0467DF?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Llama_3.2-Local_AI-7C3AED?style=for-the-badge)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](LICENSE)

**Upload any PDF → Ask questions in plain English → Get AI-powered answers**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Results](#-evaluation-results) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 What is DocMind?

DocMind is a **Retrieval Augmented Generation (RAG)** application that lets you chat with your own documents. Unlike ChatGPT which uses internet data, DocMind answers questions using **only your uploaded PDFs** — making it accurate, private, and hallucination-free.

```
Without DocMind:                    With DocMind:
─────────────────                   ──────────────────────────────
You read 50-page PDF  →  😓         Upload PDF  →  Ask question  →  ✅ Instant answer
Search manually       →  🕐         "What is my CGPA?"  →  "Your CGPA is 8.7"
Miss important info   →  ❌         Sources shown  →  Confidence score  →  🎯 Accurate
```

---

## ✨ Features

```
┌─────────────────────────────────────────────────────────────────┐
│                         DOCMIND FEATURES                        │
├─────────────────────┬───────────────────────────────────────────┤
│  🧠 Semantic Search  │  Finds meaning, not just keywords         │
│  🤖 RAG Pipeline     │  FAISS + Llama 3.2 for accurate answers   │
│  📄 PDF Upload       │  Drag & drop any PDF document             │
│  ⚡ Auto Summary     │  Instant summary on upload                │
│  📊 Confidence Score │  Shows AI confidence % for each answer    │
│  💬 Chat History     │  All conversations saved automatically    │
│  📚 Multi-Document   │  Search across multiple PDFs at once      │
│  📋 Copy Answers     │  One-click copy on every AI response      │
│  🌙 Dark/Light Mode  │  Toggle between beautiful themes          │
│  🔐 Login/Register   │  Full authentication flow                 │
│  🏠 Landing Page     │  Beautiful home page with feature tour    │
│  📊 Comparison Chart │  Proves semantic beats keyword search     │
│  🔒 100% Private     │  Your documents never leave your device   │
└─────────────────────┴───────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
                        ┌─────────────────────┐
                        │   USER INTERFACE     │
                        │   React + CSS-in-JS  │
                        │   localhost:3000     │
                        └──────────┬──────────┘
                                   │ HTTP Request
                                   ▼
                        ┌─────────────────────┐
                        │   FASTAPI BACKEND   │
                        │   Python + Uvicorn  │
                        │   localhost:8000    │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │ sentence-        │  │     FAISS       │  │   Llama 3.2      │
   │ transformers     │  │ Vector Database │  │   via Ollama     │
   │                  │  │                 │  │   (Local AI)     │
   │ Text → Vectors   │  │ Similarity      │  │                  │
   │ 384 dimensions   │  │ Search          │  │ Generates answer │
   └──────────────────┘  └─────────────────┘  └──────────────────┘

   ════════════════════════════════════════════════════════════
                     RAG PIPELINE FLOW
   ════════════════════════════════════════════════════════════

   1. PDF Upload  →  Extract text  →  Split into chunks
                                             ↓
   2. Each chunk  →  sentence-transformers  →  384-dim vector
                                             ↓
   3. All vectors  →  stored in FAISS index
                                             ↓
   4. User question  →  convert to vector  →  FAISS finds top 5 chunks
                                             ↓
   5. Top chunks + question  →  Llama 3.2  →  Final answer
                                             ↓
   6. Answer + Sources + Confidence Score  →  React UI
```

---

## 📊 Evaluation Results

> Semantic Search vs Traditional Keyword Search (TF-IDF) on 5 test queries

```
  Query                              Semantic   Keyword
  ─────────────────────────────────  ────────   ───────
  "how do machines learn?"           ✅ PASS    ❌ FAIL
  "what is meaning based search?"    ✅ PASS    ✅ PASS
  "how are words converted to nums?" ✅ PASS    ❌ FAIL
  "what finds similar vectors fast?" ✅ PASS    ✅ PASS
  "how does AI understand language?" ✅ PASS    ✅ PASS

  ┌──────────────────────────────────────────────┐
  │  SEMANTIC SEARCH  ████████████████████  100% │
  │  KEYWORD SEARCH   ████████████░░░░░░░░   60% │
  └──────────────────────────────────────────────┘

  🏆 Semantic Search wins by 40% improvement
```

---

## 🛠️ Tech Stack

```
┌─────────────┬──────────────────────────────┬─────────────────────────┐
│   Layer     │   Technology                 │   Purpose               │
├─────────────┼──────────────────────────────┼─────────────────────────┤
│  Frontend   │  React 18, CSS-in-JS         │  Chat UI, Upload, Auth  │
│  Backend    │  FastAPI, Python 3.10+       │  API server, RAG logic  │
│  Embeddings │  sentence-transformers       │  Text → Vectors         │
│             │  all-MiniLM-L6-v2           │  384-dim embeddings     │
│  Vector DB  │  FAISS (Facebook AI)         │  Similarity search      │
│  LLM        │  Llama 3.2 via Ollama        │  Answer generation      │
│  PDF        │  PyMuPDF (fitz)              │  PDF text extraction    │
│  Evaluation │  scikit-learn TF-IDF         │  Baseline comparison    │
└─────────────┴──────────────────────────────┴─────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Check Python version (needs 3.10+)
python --version

# Check Node version (needs 18+)
node --version

# Install Ollama from https://ollama.com
ollama --version
```

### Step 1 — Clone repositories

```bash
git clone https://github.com/koushika05-crypto/rag-semantic-search.git
git clone https://github.com/koushika05-crypto/rag-frontend.git
```

### Step 2 — Set up Python backend

```bash
cd rag-semantic-search

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install fastapi uvicorn sentence-transformers faiss-cpu pymupdf ollama python-multipart scikit-learn python-dotenv
```

### Step 3 — Pull Llama AI model

```bash
# This downloads ~2GB — only needed once
ollama pull llama3.2
```

### Step 4 — Start the backend server

```bash
uvicorn main:app --reload
# ✅ Running at http://127.0.0.1:8000
```

### Step 5 — Set up and start frontend

```bash
cd ../rag-frontend
npm install
npm start
# ✅ Running at http://localhost:3000
```

### Step 6 — Use the app

```
1. Open http://localhost:3000
2. Register or sign in
3. Upload any PDF from the sidebar
4. Wait for auto-summary
5. Ask any question about your document!
```

---

## 📁 Project Structure

```
rag-semantic-search/              rag-frontend/
├── main.py        ← API server  ├── src/
├── embed.py       ← Milestone 1 │   ├── App.js       ← Main chat UI
├── search.py      ← Milestone 2 │   ├── Login.js     ← Auth pages
├── rag.py         ← Milestone 3 │   ├── Compare.js   ← Chart page
├── compare.py     ← Milestone 5 │   └── index.js
├── .env           ← API keys    └── public/
├── .gitignore
└── README.md
```

---

## ✅ Project Milestones

```
Milestone 1  ██████████  Embedding Layer      Text → 384-dim vectors  ✅
Milestone 2  ██████████  FAISS Vector Store   Similarity search       ✅
Milestone 3  ██████████  RAG Pipeline         FAISS + Llama 3.2       ✅
Milestone 4  ██████████  Full-Stack App       React + FastAPI         ✅
Milestone 5  ██████████  Evaluation           Semantic vs Keyword     ✅
```

---

## 💡 How RAG Works — Simply Explained

```
Normal Search (Keyword):
  Query: "how machines learn" → looks for exact words → misses related content ❌

Semantic Search (RAG):
  Query: "how machines learn" → understands meaning → finds AI, neural networks ✅

The difference:
  Keyword  →  "Do these words match?"
  Semantic →  "Does this content mean the same thing?"
```

---

## 🤝 Author

<div align="center">

**Built by Kethireddy Koushika**

*Full-Stack Developer → AI Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-koushika05--crypto-181717?style=for-the-badge&logo=github)](https://github.com/koushika05-crypto)

*Built with React · FastAPI · FAISS · Llama 3.2 · sentence-transformers*

</div>

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

</div>