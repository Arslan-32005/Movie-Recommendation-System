# Movie Recommendation System

## Overview
Content-based movie recommendation system built on TMDB 5000 dataset. Users can get similar movie recommendations or browse movies by genre.

## Live Demo
[Movie Recommender App](https://movie-recommendation-system-czzmtcjx3sz6nmdtgtuar5.streamlit.app/)

## How It Works
- Movie metadata (genres, keywords, cast, director, overview) combined into tags
- Tags vectorized using CountVectorizer (5000 features)
- Cosine similarity calculated between all movies
- Top 5 most similar movies returned for any queried title

## Features
- Movie-based recommendations — select a movie, get 5 similar titles
- Genre-based filtering — browse movies by genre
- Real-time similarity calculation with Streamlit caching

## Dataset
- **Source:** TMDB 5000 Movies Dataset
- **Size:** 4,800+ movies
- **Features used:** title, overview, genres, keywords, cast, crew

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (CountVectorizer, Cosine Similarity)
- Streamlit
