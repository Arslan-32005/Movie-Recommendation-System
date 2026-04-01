import numpy as np
import pandas as pd
import ast
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

movies = movies.merge(credits, on="title")

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies['tags'] = movies['tags'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in movies['title'].values:
        print("Movie not found ❌")
        return
    
    index = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nTop recommendations for {movie}:\n")
    for i in movies_list:
        print(
            movies.iloc[i[0]].title, "->",
            ", ".join(movies.iloc[i[0]].genres)
        )


all_genres = []

for genre_list in movies['genres']:
    for g in genre_list:
        all_genres.append(g)

all_genres = sorted(list(set(all_genres)))


def recommend_by_genre(genre):
    filtered = movies[movies['genres'].apply(lambda x: genre in x)]
    return list(filtered['title'].head(10))


recommend('Avatar')

print("\nMovies in Science Fiction:\n")
print(recommend_by_genre("Science Fiction"))


pickle.dump(movies, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))

print("\nFiles saved successfully")