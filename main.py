from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import requests
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

model = SentenceTransformer("paraphrase-albert-small-v2")

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ========== Helpers ==========

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
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

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

def ask_gpt(question, context):
    try:
        prompt = f"Based on the following document context, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions from documents."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ========== API Endpoint ==========

@app.post("/api/v1/hackrx/run")
def handle_request(payload: HackRxRequest):
    try:
        text = extract_text_from_pdf(payload.documents)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        index = build_index(embeddings)

        answers = []
        for q in payload.questions:
            top_chunks = get_top_chunks(q, chunks, index, embeddings)
            context = " ".join(top_chunks)
            final_answer = ask_gpt(q, context)
            answers.append(final_answer)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "OrthoTrace API running. Use /api/v1/hackrx/run"}
