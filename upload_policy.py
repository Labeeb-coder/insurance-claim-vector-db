# Install required libraries first:
# pip install pdfplumber langchain chromadb sentence-transformers

import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# === CONFIGURATION ===
PDF_FILE_PATH = "/home/soorajnair/Documents/2_Cloud/TASEK_S/Task_18/T&C.pdf"     # Path to your PDF
CHROMA_DB_DIR = "/home/soorajnair/Documents/2_Cloud/TASEK_S/Task_18/chroma_db"    # Folder to persist your vector database
HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === STEP 1: Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

policy_text = extract_text_from_pdf(PDF_FILE_PATH)
print("✅ PDF text extracted. Preview (first 500 chars):\n")
print(policy_text[:500])

# === STEP 2: Split text into chunks (to handle large PDFs) ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # size of each chunk
    chunk_overlap=50   # overlap between chunks
)
texts = splitter.split_text(policy_text)
print(f"✅ Text split into {len(texts)} chunks for embedding.")

# === STEP 3: Initialize Hugging Face embeddings ===
embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)

# === STEP 4: Initialize Chroma DB ===
vector_db = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

# === STEP 5: Add chunks to Chroma DB ===
vector_db.add_texts(
    texts=texts,
    metadatas=[{"source": PDF_FILE_PATH}] * len(texts)
)
vector_db.persist()

print(f"✅ {len(texts)} chunks stored in Chroma DB at '{CHROMA_DB_DIR}'!")
