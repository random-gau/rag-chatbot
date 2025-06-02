import os
import time
import tempfile

# Fix for torch error in some systems
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Constants
MODEL_PATH = "Models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot (Mistral Local)")
st.write("Ask anything about your uploaded document.")

# Upload PDF
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type="pdf")

# Process PDF
if uploaded_pdf:
    with st.spinner("📚 Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.session_state.db = FAISS.from_documents(chunks, embedding_model)

        st.success("✅ PDF processed and indexed successfully!")

# Load LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=4096,
    verbose=False
)

# Clear chat history
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Chat input
query = st.text_input("💬 Enter your question here")
if st.button("Ask") and query:
    if not st.session_state.db:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        with st.spinner("⏳ Thinking..."):
            retriever = st.session_state.db.as_retriever(search_kwargs={"k": 4})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            start_time = time.time()
            response = qa_chain.invoke(query)
            end_time = time.time()

            answer = response['result']
            st.session_state.chat_history.append((query, answer))
            st.markdown(f"**🤖 Answer:** {answer}")
            st.markdown(f"⏱️ _Answered in {end_time - start_time:.2f} seconds._")

# Display chat history
if st.session_state.chat_history:
    st.subheader("🧾 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**🤖 Answer:** {a}")
        st.markdown("---")
