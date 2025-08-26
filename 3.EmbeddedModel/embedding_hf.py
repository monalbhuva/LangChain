from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)


text = "Hugging Face makes working with transformers easy!"
vector = embeddings.embed_query(text)

print("Embedding length:", len(vector))
print("First 10 values:", vector[:10])
