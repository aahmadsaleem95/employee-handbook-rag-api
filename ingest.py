import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import re

# Simple regex-based sentence tokenizer
def sent_tokenize(text: str):
    # Split on sentence-ending punctuation followed by whitespace or newline
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection("handbook")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process the PDF
doc = fitz.open("employee_handbook.pdf")
chunk_id = 0

for page_num, page in enumerate(doc):
    text = page.get_text().strip()
    if not text:
        continue

    sentences = sent_tokenize(text)

    # Combine every 2 sentences (or keep 1 if only 1 remains)
    chunks = [" ".join(sentences[i:i + 2]) for i in range(0, len(sentences), 2)]

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{page_num}_{chunk_id}"]
        )
        chunk_id += 1

print(f"Stored {chunk_id} regex-based chunks ✅")
