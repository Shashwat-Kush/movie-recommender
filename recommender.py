import streamlit as st
import pickle
import requests
from config import (
    MOVIES_PICKLE_PATH, 
    SIMILARITY_PICKLE_PATH,
    TMDB_API_URL,
    TMDB_API_KEY,
    TMDB_POSTER_BASE_URL,
    API_TIMEOUT,
    NUM_RECOMMENDATIONS,
    PLACEHOLDER_POSTER
)

# ==================== DATA LOADING ====================

@st.cache_resource
def load_data():
    """Load movies and similarity matrix from pickle files"""
    movies = pickle.load(open(MOVIES_PICKLE_PATH, 'rb'))
    similarity = pickle.load(open(SIMILARITY_PICKLE_PATH, 'rb'))
    return movies, similarity

# ==================== API CALLS ====================

@st.cache_data
def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API with caching"""
    try:
        response = requests.get(
            f'{TMDB_API_URL}/{movie_id}?api_key={TMDB_API_KEY}&language=en-US',
            timeout=API_TIMEOUT
        )
        data = response.json()
        return f"{TMDB_POSTER_BASE_URL}/{data['poster_path']}"
    except:
        return PLACEHOLDER_POSTER

# ==================== DATA FILTERING ====================

def get_all_genres(movies_list):
    """Extract all unique genres from movies"""
    all_genres = set()
    for genres_list in movies_list['genres']:
        all_genres.update(genres_list)
    return sorted(list(all_genres))

def filter_movies_by_genre(movies_list, selected_genres):
    """Filter movies based on selected genres"""
    if not selected_genres:
        return movies_list.copy()
    return movies_list[movies_list['genres'].apply(lambda x: any(g in selected_genres for g in x))]

# ==================== RECOMMENDATION ENGINE ====================

def get_recommendations(movies_list, similarity, movie_name):
    """Get 5 movie recommendations based on similarity"""
    movie_index = movies_list[movies_list['title'] == movie_name].index[0]
    distances = similarity[movie_index]
    movies_indices = sorted(
        list(enumerate(distances)), 
        reverse=True, 
        key=lambda x: x[1]
    )[1:NUM_RECOMMENDATIONS+1]
    
    recommendations = {
        'names': [],
        'posters': [],
        'genres': [],
        'overviews': []
    }
    
    for i in movies_indices:
        idx = i[0]
        recommendations['names'].append(movies_list.iloc[idx]['title'])
        recommendations['posters'].append(fetch_poster(movies_list.iloc[idx]['movie_id']))
        recommendations['genres'].append(movies_list.iloc[idx]['genres'])
        recommendations['overviews'].append(movies_list.iloc[idx]['overview'])
    
    return recommendations
