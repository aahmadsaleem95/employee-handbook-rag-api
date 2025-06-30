# rag.py
from sentence_transformers import SentenceTransformer, util
import chromadb
import ollama

# Load persistent Chroma vector store
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_collection("handbook")

# Embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("sentence-t5-base")
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# 3. Expand user question into multiple semantic variants
def expand_question_variants(question: str):
    base = question.strip().rstrip("?")
    return list(set([
        question,
        f"What is the policy on {base}?",
        f"Can you explain {base}?",
        f"Give me details about {base}.",
        f"What does the employee handbook say about {base}?",
        f"When does {base} happen?",
        f"On what day is {base} done?"
    ]))

def get_answer(question: str) -> str:
    queries = expand_question_variants(question)
    all_candidates = []

    # For each variant: embed → query → collect
    for q in queries:
        q_embedding = model.encode(q).tolist()
        results = collection.query(query_embeddings=[q_embedding], n_results=3)
        docs = results["documents"][0]
        all_candidates.extend(docs)

    # Deduplicate and rerank
    unique_docs = list(set(all_candidates))
    reranked = sorted(
        [(doc, util.cos_sim(model.encode(question), model.encode(doc)).item()) for doc in unique_docs],
        key=lambda x: x[1],
        reverse=True
    )

    # Select top 2 chunks
    top_chunks = [doc for doc, _ in reranked[:2]]
    context = "\n\n".join(top_chunks)

    # Build final prompt
    prompt = f"""You are a helpful HR assistant. Use only the context provided to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:"""

    # Call Ollama with gemma:2b
    response = ollama.chat(
        model="gemma:2b",
        options={"num_predict": 200},
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
