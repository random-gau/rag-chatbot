import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Step 1: Load the FAISS vector store
print("🔧 Loading vector store...")
start = time.time()
db = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)
print(f"✅ Vector store loaded in {time.time() - start:.2f} seconds.")

# Step 2: Load the local LLM
print("🧠 Loading local LLM...")
start = time.time()
llm = LlamaCpp(
    model_path="D:/Projects/rag-chatbot/Models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    top_p=1,
    n_ctx=4096,
    verbose=False,
)
print(f"✅ Local LLM loaded in {time.time() - start:.2f} seconds.")

# Step 3: Setup the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# Step 4: Run the chat loop
print("🤖 RAG Chatbot ready. Ask a question (type 'exit' to quit).")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    print("⏳ Processing...")
    start = time.time()
    answer = qa_chain.run(query)
    print(f"🤖: {answer} (answered in {time.time() - start:.2f} seconds)\n")
