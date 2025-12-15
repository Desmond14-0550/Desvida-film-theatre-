import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

movies = [
    {"title": "Inception", "overview": "A thief who steals secrets through dreams."},
    {"title": "Interstellar", "overview": "A space mission to save humanity."},
    {"title": "The Dark Knight", "overview": "Batman faces the Joker."}
]

docs = [m["title"] + " " + m["overview"] for m in movies]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(docs)

st.title("🎬 Movie Search Engine")

query = st.text_input("Search for a movie")

if query:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    top = sims.argsort()[-5:][::-1]

    for i in top:
        st.subheader(movies[i]["title"])
        st.write(movies[i]["overview"])
      
