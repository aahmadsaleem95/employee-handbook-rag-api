# rag.py
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Connect to persistent Chroma store
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_collection("handbook")

# Embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_answer(question: str) -> str:
    q_embedding = model.encode(question).tolist()
    
    # Fetch top 5 most relevant document chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3
    )
    
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are an assistant answering HR-related questions using the employee handbook.

Answer the following question using only the context below. Be accurate, concise, and formal.

Context:
{context}

Question:
{question}

Answer:"""

    response = ollama.chat(
        model="gemma:2b",  # much faster than mistral
        options={"num_predict": 150},
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
