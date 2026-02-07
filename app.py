import streamlit as st
from config import PAGE_LAYOUT, PAGE_TITLE, PAGE_ICON
from recommender import (
    load_data, 
    get_all_genres, 
    filter_movies_by_genre, 
    get_recommendations
)
from ui import render_header, render_sidebar, render_recommendations

# ==================== PAGE CONFIG ====================

st.set_page_config(layout=PAGE_LAYOUT, page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# ==================== LOAD DATA ====================

movies_list, similarity = load_data()

# ==================== MAIN APP ====================

def main():
    """Main application logic"""
    render_header()
    
    # Get sidebar filters
    all_genres = get_all_genres(movies_list)
    selected_genres = render_sidebar(all_genres)
    
    # Filter and select movie
    filtered_movies = filter_movies_by_genre(movies_list, selected_genres)
    selected_movie_name = st.selectbox('Select a movie:', filtered_movies['title'].values)
    
    # Get recommendations on button click
    if st.button('🎯 Get Recommendations', use_container_width=True):
        recommendations = get_recommendations(movies_list, similarity, selected_movie_name)
        render_recommendations(recommendations)

if __name__ == "__main__":
    main()

