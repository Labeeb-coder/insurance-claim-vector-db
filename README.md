🛡️ Insurance Policy Vector Database (LangChain + Chroma + HuggingFace)
📘 Overview

This project is designed to extract, process, and store insurance policy documents into a vector database for advanced semantic search and analysis.
By integrating LangChain, HuggingFace Embeddings, and ChromaDB, this system enables intelligent retrieval and validation of insurance terms and conditions — a foundation for building AI-based claim validation or RAG (Retrieval-Augmented Generation) systems.

⚙️ Key Features

PDF Text Extraction using pdfplumber

Chunking & Embedding using LangChain and HuggingFace

Semantic Vector Storage with ChromaDB

Efficient Query Handling for fast policy lookups

Scalable Design ready for integration with LLMs

🧠 Tech Stack
Category	Technology
Programming Language	Python 3.10+
Framework	LangChain
Embedding Model	sentence-transformers/all-MiniLM-L6-v2
Vector Store	ChromaDB
PDF Processing	pdfplumber
Model Hosting	HuggingFace Transformers
🏗️ Project Structure
insurance-claim-vector-db/
│
├── upload_policy.py          # Main script for PDF extraction and vector upload
├── T&C.pdf                   # Example Insurance Policy Document
├── chroma.sqlite3            # Persisted Chroma Vector Database
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation

🚀 Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/Labeeb-coder/insurance-claim-vector-db.git
cd insurance-claim-vector-db

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Script
python upload_policy.py

📦 How It Works

Extracts text content from the insurance policy PDF

Splits the text into semantic chunks for efficient embedding

Converts text into vector representations using a pretrained transformer model

Persists these vectors in ChromaDB for semantic search and retrieval

Example Output:

✅ PDF text extracted. Preview (first 500 chars):
...
✅ Text split into 128 chunks for embedding.
✅ 128 chunks stored in Chroma DB at 'chroma_db/'!

💡 Use Case Scenarios

🧾 AI Claim Validation System — compare claim details with policy terms

🔍 Document Q&A Interface — retrieve policy clauses using natural language

🧠 RAG Integration — combine vector retrieval with LLMs like GPT or LLaMA

🧩 Insurance Chatbot Backend — build context-aware insurance support bots

📁 Requirements
pdfplumber
langchain
chromadb
sentence-transformers

👨‍💻 Author

Muhammed Labeeb
AI Intern | Aspiring AI Developer | Passionate about Generative AI
📍 India
💬 “Ideas don't have value, only practical ideas have value.”

📜 License

This project is licensed under the MIT License — free to use, modify, and distribute with proper credit.
