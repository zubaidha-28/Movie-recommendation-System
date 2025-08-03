import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
movies = pd.read_csv("movies.csv")
movies['overview'] = movies['overview'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation logic
def recommend(title, num_recommendations=5):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_name = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    st.write("**Recommended Movies:**")
    for i, rec in enumerate(recommend(movie_name), 1):
        st.write(f"{i}. {rec}")
