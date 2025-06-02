# build_vectorstore.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 1. Load your PDF
pdf_path = "data/your_file.pdf"  # Replace with your actual PDF file path
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Embed using a local HuggingFace model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create vectorstore
vectorstore = FAISS.from_documents(docs, embedding_model)

# 5. Save the vectorstore to disk
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore")

print("✅ Vectorstore created and saved to 'vectorstore/'")
