# 🎬 Movie Recommender System

A content-based movie recommendation system built with machine learning and Streamlit.

## Features

- 🎯 **Smart Recommendations** - Get 5 personalized movie recommendations based on your favorite film
- 🎭 **Genre Filtering** - Filter movies by genre before selecting
- 🖼️ **Movie Posters** - View beautiful poster images from TMDB API
- 📝 **Plot Summaries** - Read full movie overviews
- ⚡ **Fast & Cached** - Optimized with Streamlit caching for quick responses

## How It Works

The system uses **content-based filtering** with cosine similarity to recommend movies based on:
- Movie genres
- Cast members
- Directors
- Keywords
- Plot descriptions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shashwat-Kush/movie-recommender.git
cd movie-recommender
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the datasets and pickle files (required):
- `movies.pkl` - Processed movies dataframe
- `similarity.pkl` - Cosine similarity matrix
- `tmdb_5000_movies.csv` - Original TMDB dataset
- `tmdb_5000_credits.csv` - TMDB credits dataset

## Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
movie-recommender/
├── app.py              # Main Streamlit application
├── config.py           # Configuration & constants
├── recommender.py      # Recommendation engine & data loading
├── ui.py               # UI components
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── movies.pkl          # Pickled movies dataframe (gitignored)
└── similarity.pkl      # Pickled similarity matrix (gitignored)
```

## Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web framework
- **Pandas** - Data processing
- **Scikit-learn** - Machine learning (cosine similarity)
- **NLTK** - Natural language processing
- **Requests** - API calls to TMDB

## Dataset

Uses the TMDB 5000 Movies dataset containing:
- 5000 movies
- Genres, cast, crew, keywords
- Movie overviews and ratings

## API

Fetches movie posters from [The Movie Database (TMDB) API](https://www.themoviedb.org/settings/api)

## License

MIT License - feel free to use this project for learning and personal use.

## Author

[Shashwat Kush](https://github.com/Shashwat-Kush)
