from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_store(folder_path):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store

def main():
    vector_store = load_vector_store("faiss_index")
    print("Vector store loaded successfully!")

if __name__ == "__main__":
    main()
