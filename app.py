import streamlit as st
import os

st.write("Secrets folder exists?", os.path.exists(".streamlit"))
st.write("Secrets file exists?", os.path.exists(".streamlit/secrets.toml"))

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.write("API Key:", OPENAI_API_KEY[:5] + "...")
except Exception as e:
    st.error(f"Error reading secrets: {e}")

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Get OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

def main():
    st.title("🦜🔗 RAG Chatbot with Streamlit")

    vector_store = load_vectorstore()
    llm = load_llm()

    # Create a RetrievalQA chain that uses vector store retriever + LLM
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    user_question = st.text_input("Ask your question about the documents:")

    if user_question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_question)
        st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
