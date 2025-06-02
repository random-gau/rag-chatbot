import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import time
import os

# Load LLM
llm = LlamaCpp(
    model_path="Models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    top_p=1,
    n_ctx=4096,
    verbose=False,
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("📚 RAG Chatbot (Mistral Local)")
st.write("Upload a PDF and ask anything about it.")

# File upload
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract text from PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # RAG chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    st.success("✅ PDF processed. Ask your question below!")

    # Question input
    query = st.text_input("🔍 Ask a question about the document")

    if query:
        start = time.time()
        with st.spinner("⏳ Processing..."):
            response = qa.run(query)
        end = time.time()
        st.markdown(f"**Answer:** {response}")
        st.caption(f"⏱️ Answered in {end - start:.2f} seconds.")
else:
    st.info("Please upload a PDF to get started.")
