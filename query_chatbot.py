from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_store(folder_path):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store

def query_chatbot(vector_store, query_text, top_k=3):
    results = vector_store.similarity_search(query_text, k=top_k)
    answers = [doc.page_content for doc in results]
    return "\n---\n".join(answers)

def main():
    print("Loading vector store...")
    vector_store = load_vector_store("faiss_index")
    print("Vector store loaded successfully!")

    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = query_chatbot(vector_store, user_input)
        print(f"Bot:\n{response}")

if __name__ == "__main__":
    main()
