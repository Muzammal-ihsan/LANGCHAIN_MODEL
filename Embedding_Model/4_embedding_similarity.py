import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create embedding model
embedding_model = genai.GenerativeModel("embedding-001")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

# Embed documents
doc_embeddings = [
    genai.embed_content(model="embedding-001", content=doc)["embedding"]
    for doc in documents
]

# Embed query
query_embedding = genai.embed_content(model="embedding-001", content=query)["embedding"]

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query:", query)
print("Best Match:", documents[index])
print("Similarity Score:", score)
