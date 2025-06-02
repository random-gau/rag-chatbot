from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model WITHOUT token argument
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "This is a test sentence."
embeddings = embedding_model.embed_query(text)

print(embeddings)
