import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Dummy pivot table
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
similarity = cosine_similarity(user_movie_matrix)

st.title("🎬 Movie Recommender")

user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['userId'].max())

if st.button("Recommend"):
    if user_id in user_movie_matrix.index:
        sim_scores = list(enumerate(similarity[user_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        st.subheader("Top Recommendations:")
        for i, score in sim_scores:
            st.write(f"User {i} - Similarity Score: {score:.2f}")
    else:
        st.error("User not found!")
