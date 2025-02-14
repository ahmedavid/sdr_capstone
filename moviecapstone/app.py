import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

from index_generator import create_textual_representation

df = pd.read_csv('netflix_titles.csv')


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


index_path = "index"
index = faiss.read_index(index_path)

if 'textual_rep' not in df.columns:
    df['textual_rep'] = df.apply(create_textual_representation, axis=1)


st.title("🎬 Movie Recommendation ")
st.write("Search for movies by title or describe the kind of movie you want!")


search_option = st.radio("How do you want to search?", ["🔍 Search by Movie", "✍ Describe a Movie"])

if search_option == "🔍 Search by Movie":
    movie_name = st.text_input("Enter a movie name:")

    if st.button("Find Similar Movies") and movie_name:

        user_embedding = embedding_model.encode(movie_name, convert_to_numpy=True)
        embedding = np.array([user_embedding], dtype='float32')

        
        D, I = index.search(embedding, 5)

   
        best_matches = np.array(df['textual_rep'])[I.flatten()]

       
        st.subheader("🎬 Recommended Movies:")
        for match in best_matches:
            
            match_details = df[df['textual_rep'] == match].iloc[0]
            st.write(f"🍿 **{match_details['title']}**")
            st.write(f"🎭 **Genres:** {match_details['listed_in']}")
            st.write(f"🎬 **Director:** {match_details['director']}")
            st.write(f"👥 **Cast:** {match_details['cast']}")
            st.write(f"📅 **Release Year:** {match_details['release_year']}")
            st.write(f"📝 **Description:** {match_details['description']}")

elif search_option == "✍ Describe a Movie":
    user_description = st.text_area("Describe the kind of movie you want:")

    if st.button("Find Similar Movies") and user_description:
        
        user_embedding = embedding_model.encode(user_description, convert_to_numpy=True)
        embedding = np.array([user_embedding], dtype='float32')

       
        D, I = index.search(embedding, 5)

        best_matches = np.array(df['textual_rep'])[I.flatten()]

        
        st.subheader("🎬 Recommended Movies:")
        for match in best_matches:
            
            match_details = df[df['textual_rep'] == match].iloc[0]
            st.write(f"🍿 **{match_details['title']}**")
            st.write(f"🎭 **Genres:** {match_details['listed_in']}")
            st.write(f"🎬 **Director:** {match_details['director']}")
            st.write(f"👥 **Cast:** {match_details['cast']}")
            st.write(f"📅 **Release Year:** {match_details['release_year']}")
            st.write(f"📝 **Description:** {match_details['description']}")
