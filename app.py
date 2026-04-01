import streamlit as st
import pickle
import pandas as pd


movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))


all_genres = []
for genre_list in movies['genres']:
    for g in genre_list:
        all_genres.append(g)
all_genres = sorted(list(set(all_genres)))


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(
            movies.iloc[i[0]].title + " -> " + ", ".join(movies.iloc[i[0]].genres)
        )
    return recommended_movies


def recommend_by_genre(genre):
    filtered = movies[movies['genres'].apply(lambda x: genre in x)]
    return list(filtered['title'].head(10))


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender System")


tab1, tab2 = st.tabs(["Movie Based", "Genre Based"])


with tab1:
    st.subheader("Get similar movies based on a movie")
    selected_movie = st.selectbox("Select a Movie", movies['title'].values)
    if st.button("Recommend Movie", key="movie_button"):
        recommendations = recommend(selected_movie)
        st.write("Top 5 similar movies:")
        for rec in recommendations:
            st.write(rec)

with tab2:
    st.subheader("Find movies by genre")
    selected_genre = st.selectbox("Select a Genre", all_genres)
    if st.button("Show Movies by Genre", key="genre_button"):
        genre_movies = recommend_by_genre(selected_genre)
        st.write(f"Movies in genre: {selected_genre}")
        for m in genre_movies:
            st.write(m)
