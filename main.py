from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import fitz
import requests
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== INPUT FORMAT ==========
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ========== HELPER FUNCTIONS ==========

def extract_text_from_pdf(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

def chunk_text(text, max_words=150):
    words = text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def get_top_chunks(question, chunks, index, embeddings, top_k=3):
    query_vec = model.encode([question])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

# ========== MAIN API ==========
@app.post("/api/v1/hackrx/run")
def handle_request(payload: HackRxRequest):
    try:
        # Step 1: Extract and chunk text
        text = extract_text_from_pdf(payload.documents)
        chunks = chunk_text(text)

        # Step 2: Embed chunks
        embeddings = embed_chunks(chunks)
        index = build_index(embeddings)

        # Step 3: For each question, get top chunks and simulate an answer
        answers = []
        for q in payload.questions:
            top_chunks = get_top_chunks(q, chunks, index, embeddings)
            combined = " ".join(top_chunks)
            # Simulate LLM answer (just sending back top chunks)
            answers.append(combined.strip())

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
