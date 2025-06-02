import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --- Step 1: Extract text from PDF ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Step 2: Split text into chunks ---
def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# --- Step 3: Create embeddings and store in FAISS vector store ---
def create_vector_store(chunks):
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store from chunks
    vector_store = FAISS.from_texts(chunks, embedding_model)
    return vector_store

def main(pdf_path):
    # Extract text
    print("Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)

    # Split text
    print("Splitting text into chunks...")
    chunks = split_text(full_text)
    print(f"Total chunks created: {len(chunks)}")

    # Create vector store
    print("Creating vector store with embeddings...")
    vector_store = create_vector_store(chunks)

    # Save the vector store locally
    vector_store.save_local("faiss_index")

    print("Vector store saved locally as 'faiss_index' folder.")

if __name__ == "__main__":
    pdf_file_path = "ai-report.pdf"  # Replace with your PDF file path
    main(pdf_file_path)
