from llama_cpp import Llama

# ✅ Update this to your actual model path
model_path = "D:/Projects/rag-chatbot/Models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Load the model
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=6)

# Prompt the model
prompt = "What is a RAG chatbot?"
output = llm(prompt, max_tokens=200)

# Print the result
print("\n=== Model Response ===\n")
print(output["choices"][0]["text"].strip())
from llama_cpp import Llama

llm = Llama(model_path="Models/mistral-7b/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

response = llm("Q: What is the capital of France?\nA:", max_tokens=50)
print(response["choices"][0]["text"])
