import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os


df = pd.read_csv('netflix_titles.csv')

#  Create textual representation for embeddings
def create_textual_representation(row):
    textual_representation = f"""Type: {row['type']},
Title: {row['title']},
Director: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},

Description: {row['description']}"""
    return textual_representation

df['textual_rep'] = df.apply(create_textual_representation, axis=1)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

dim = 384  
index_path = "index"

if not os.path.exists(index_path):  
    print("⚡ Generating FAISS index...")
    index = faiss.IndexFlatL2(dim)
    X = np.zeros((len(df['textual_rep']), dim), dtype='float32')

    # Compute embeddings
    for i, representation in enumerate(df['textual_rep']):
        if i % 30 == 0:
            print(f'Processed {i} instances')

        embedding = embedding_model.encode(representation, convert_to_numpy=True)
        X[i] = embedding  

    # Add embeddings to FAISS
    index.add(X)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print("✅ FAISS index saved!")
else:
    print("✅ FAISS index already exists, skipping computation.")
