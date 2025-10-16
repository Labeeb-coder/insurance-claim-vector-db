import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import Chroma as BaseChroma
from dotenv import load_dotenv
import os
from groq import Groq

# ---------- Load environment variables ----------
load_dotenv()  # loads variables from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ---------- Config ----------
GROQ_MODEL = "llama-3.3-70b-versatile"  # replace with recommended Groq model
CHROMA_DB_DIR = "./chroma_db"
HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5  # number of relevant chunks to retrieve

# ---------- Functions ----------

def extract_text_from_pdf(file):
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split long text into smaller overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def embed_and_store(chunks):
    """Embed chunks using HuggingFace model and store in Chroma vector DB."""
    embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    vector_db.add_texts(chunks)
    vector_db.persist()
    return vector_db

def retrieve_similar_chunks(text, top_k=TOP_K):
    """Retrieve top-k similar chunks from Chroma DB."""
    embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)
    vector_db = BaseChroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    if vector_db._collection.count() == 0:
        return []  # No previous claims stored
    results = vector_db.similarity_search(text, k=top_k)
    return [r.page_content for r in results]

def validate_claim_with_groq_api(new_text, query="Is this insurance claim valid?"):
    """Validate claim using Groq API with historical context from Chroma DB."""
    # Retrieve historical similar claims
    historical_chunks = retrieve_similar_chunks(new_text)
    context_text = "\n\n".join(historical_chunks) if historical_chunks else "No historical claims found."

    prompt = f"{query}\n\nNew Claim:\n{new_text}\n\nHistorical Claims Context:\n{context_text}"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error calling Groq API: {e}"
# ---------- Streamlit UI ----------
st.title("🛡️ AI-Based Insurance Claim Validator (RAG + Groq LLM)")
st.write("Upload a PDF claim document to analyze and validate using Groq LLM with historical claims stored in Chroma DB.")

uploaded_file = st.file_uploader("📄 Upload Insurance Claim PDF", type=["pdf"])

if uploaded_file:
    st.info("🔍 Extracting text from PDF...")
    text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("❌ No text found in PDF.")
    else:
        st.success("✅ Text extracted successfully!")

        st.info("✂️ Splitting text into chunks...")
        chunks = split_text(text)
        st.write(f"Text split into **{len(chunks)} chunks**.")

        st.info("🧠 Embedding and storing chunks in Chroma DB...")
        embed_and_store(chunks)
        st.success("✅ Data embedded successfully!")

        st.info("🤖 Validating claim using Groq API with historical context...")
        result = validate_claim_with_groq_api(text)

        # Display the full result first
        st.subheader("📄 Claim Analysis:")
        st.write(result)

        # Highlight the validity decision
        # We'll look for keywords in the result to determine Valid / Not Valid
        lower_result = result.lower()
        if "valid" in lower_result and "not valid" not in lower_result:
            validity = "VALID ✅"
            color = "green"
        elif "not valid" in lower_result or "invalid" in lower_result:
            validity = "NOT VALID ❌"
            color = "red"
        else:
            validity = "UNCERTAIN ⚠️"
            color = "orange"

        st.markdown(f"""
            <div style="border:2px solid {color}; padding:15px; border-radius:10px; font-size:28px; font-weight:bold; text-align:center;">
                {validity}
            </div>
        """, unsafe_allow_html=True)
