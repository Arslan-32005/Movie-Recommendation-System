import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    credits = pd.read_csv('credits.csv')
    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def convert_cast(text):
        return [i['name'] for i in ast.literal_eval(text)][:3]

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join([i.replace(" ", "") for i in x]))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

all_genres = sorted(set(g for genre_list in movies['genres'] for g in genre_list))

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title + " -> " + ", ".join(movies.iloc[i[0]].genres) for i in distances]

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