# 🛡️ **INSURANCE POLICY VECTOR DATABASE**
### *(LangChain + Chroma + HuggingFace Integration)*

---

## 📘 **OVERVIEW**
This project provides an **AI-powered backend system** for processing and storing **insurance policy documents** into a **semantic vector database**.  
It leverages **LangChain**, **HuggingFace Embeddings**, and **ChromaDB** to enable **intelligent retrieval**, **context-aware validation**, and **question-answering** based on policy terms — forming the foundation for an **AI-driven insurance claim validation system** or **RAG (Retrieval-Augmented Generation)** pipeline.

---

## ⚙️ **KEY FEATURES**
✅ **PDF Text Extraction** — Parse structured text from policy PDFs using `pdfplumber`  
✅ **Chunking & Embedding** — Break down large documents into semantic segments with `LangChain`  
✅ **Vector Storage with ChromaDB** — Persist embeddings for fast similarity search  
✅ **Query-Ready Design** — Enables efficient retrieval for claim verification or LLM integration  
✅ **Scalable Architecture** — Easily extendable to multi-document RAG pipelines or chatbots  

---

## 🧠 **TECH STACK**

| **Category** | **Technology Used** |
|---------------|---------------------|
| Programming Language | Python 3.10+ |
| Framework | LangChain |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| PDF Processing | pdfplumber |
| Model Hosting | HuggingFace Transformers |

---

## 🏗️ **PROJECT STRUCTURE**
insurance-claim-vector-db/
│
├── upload_policy.py # Main script for PDF extraction & vector upload
├── T&C.pdf # Example Insurance Policy Document
├── chroma.sqlite3 # Persisted Chroma Vector Database
├── requirements.txt # Required Python packages
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 **SETUP & INSTALLATION**

### 🧩 Step 1 — Clone the Repository
```bash
git clone https://github.com/Labeeb-coder/insurance-claim-vector-db.git
cd insurance-claim-vector-db
⚙️ Step 2 — Install Dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Step 3 — Run the Script
bash
Copy code
python upload_policy.py
📦 HOW IT WORKS
1️⃣ Extracts raw text from the uploaded insurance policy PDF
2️⃣ Splits text into semantic chunks using RecursiveCharacterTextSplitter
3️⃣ Generates vector embeddings with a pretrained transformer model
4️⃣ Stores all vectors persistently in ChromaDB for retrieval
5️⃣ Ready for integration with RAG systems or claim analysis AI models

Example Output:

vbnet
Copy code
✅ PDF text extracted. Preview (first 500 chars):
...
✅ Text split into 128 chunks for embedding.
✅ 128 chunks stored in Chroma DB at 'chroma_db/'!
💡 POTENTIAL USE CASES
🔹 AI Claim Validation System — Cross-check claims with policy clauses automatically
🔹 Intelligent Q&A System — Retrieve exact policy terms using natural language
🔹 RAG + LLM Integration — Enhance model context for insurance-related queries
🔹 Insurance Chatbot Backend — Build contextual, knowledge-driven assistants

📁 REQUIREMENTS
nginx
Copy code
pdfplumber
langchain
chromadb
sentence-transformers
🔮 FUTURE ENHANCEMENTS
✨ Integration with Groq LLM or OpenAI API for contextual analysis
🧩 Build a web-based upload & query dashboard using Streamlit or FastAPI
📊 Develop an AI Claim Scoring System for quick eligibility checks
☁️ Deploy on cloud (AWS / Azure / GCP) with Docker containers

👨‍💻 AUTHOR
Muhammed Labeeb
AI Intern | Aspiring AI Developer | Passionate about Generative AI
📍 India
💬 “Ideas don't have value, only practical ideas have value.”

📜 LICENSE
This project is licensed under the MIT License — free to use, modify, and distribute with attribution.
