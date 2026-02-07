import streamlit as st

# ==================== UI COMPONENTS ====================

def render_movie_card(name, poster, genres, overview, col):
    """Render a single movie card"""
    with col:
        with st.container(border=True):
            st.image(poster, use_container_width=True)
            st.subheader(name)
            st.write(f"🎭 **Genres:** {', '.join(genres)}")
            with st.expander("See plot"):
                st.write(' '.join(overview))

def render_recommendations(recommendations):
    """Render all recommendation cards in a grid layout"""
    st.markdown("---")
    st.subheader("Recommended Movies")
    
    # First row: 3 columns
    col1, col2, col3 = st.columns(3)
    render_movie_card(
        recommendations['names'][0], 
        recommendations['posters'][0], 
        recommendations['genres'][0], 
        recommendations['overviews'][0], 
        col1
    )
    render_movie_card(
        recommendations['names'][1], 
        recommendations['posters'][1], 
        recommendations['genres'][1], 
        recommendations['overviews'][1], 
        col2
    )
    render_movie_card(
        recommendations['names'][2], 
        recommendations['posters'][2], 
        recommendations['genres'][2], 
        recommendations['overviews'][2], 
        col3
    )
    
    # Second row: 2 columns
    col4, col5 = st.columns(2)
    render_movie_card(
        recommendations['names'][3], 
        recommendations['posters'][3], 
        recommendations['genres'][3], 
        recommendations['overviews'][3], 
        col4
    )
    render_movie_card(
        recommendations['names'][4], 
        recommendations['posters'][4], 
        recommendations['genres'][4], 
        recommendations['overviews'][4], 
        col5
    )

def render_sidebar(all_genres):
    """Render sidebar filters and return selected genres"""
    st.sidebar.header("🎬 Filters")
    selected_genres = st.sidebar.multiselect(
        'Filter by genre:',
        all_genres,
        help="Select genres to filter movies"
    )
    return selected_genres

def render_header():
    """Render page header"""
    st.title('🎬 Movie Recommender System')
    st.markdown("Find movies similar to your favorite films")
