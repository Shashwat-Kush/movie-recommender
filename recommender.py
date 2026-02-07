import streamlit as st
import pickle
import requests
import os
import ast
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pandas as pd

# ==================== DATA LOADING ====================

@st.cache_resource
def load_data():
    """Load movies and similarity matrix from pickle files or generate from CSV"""
    
    # Check if pickle files exist
    if os.path.exists(MOVIES_PICKLE_PATH) and os.path.exists(SIMILARITY_PICKLE_PATH):
        movies = pickle.load(open(MOVIES_PICKLE_PATH, 'rb'))
        similarity = pickle.load(open(SIMILARITY_PICKLE_PATH, 'rb'))
        return movies, similarity
    
    # If not, generate from CSV files
    st.info("Generating recommendation model from datasets...")
    
    try:
        # Load CSV data
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge datasets
        movies = movies.merge(credits, on='title')
        
        # Select relevant columns
        movies = movies[['movie_id', 'title', 'keywords', 'genres', 'overview', 'cast', 'crew']]
        
        # Remove nulls
        movies.dropna(inplace=True)
        
        # Convert string representations to lists
        def convert(obj):
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        
        def convert3(obj):
            count = 0
            L = []
            for i in ast.literal_eval(obj):
                if count != 3:
                    L.append(i['name'])
                    count += 1
                else:
                    break
            return L
        
        def fetch_director(obj):
            L = []
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
            return L
        
        movies['genres'] = movies['genres'].apply(convert)
        movies['keywords'] = movies['keywords'].apply(convert)
        movies['cast'] = movies['cast'].apply(convert3)
        movies['crew'] = movies['crew'].apply(fetch_director)
        
        # Process text
        movies['overview'] = movies['overview'].apply(lambda x: x.split())
        
        movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
        
        # Create tags
        movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        
        # Final dataframe
        new_df = movies[['movie_id', 'title', 'tag', 'genres', 'overview']]
        new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))
        new_df['tag'] = new_df['tag'].apply(lambda x: x.lower())
        
        # Stemming
        ps = PorterStemmer()
        def stem(text):
            y = []
            for i in text.split():
                y.append(ps.stem(i))
            return " ".join(y)
        
        new_df['tag'] = new_df['tag'].apply(stem)
        
        # Vectorize
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(new_df['tag']).toarray()
        
        # Calculate similarity
        similarity = cosine_similarity(vectors)
        
        # Save pickle files
        pickle.dump(new_df, open(MOVIES_PICKLE_PATH, 'wb'))
        pickle.dump(similarity, open(SIMILARITY_PICKLE_PATH, 'wb'))
        
        st.success("Model generated successfully!")
        return new_df, similarity
        
    except FileNotFoundError:
        st.error("CSV files not found. Please upload tmdb_5000_movies.csv and tmdb_5000_credits.csv")
        return None, None

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
