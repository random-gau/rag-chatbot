import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load .env variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 1. Load PDFs from 'data' folder
pdf_path = "data"
loaders = [PyPDFLoader(os.path.join(pdf_path, f)) for f in os.listdir(pdf_path) if f.endswith(".pdf")]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# 3. Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in FAISS vector DB
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index")

print("✅ PDF processed and FAISS index created.")
