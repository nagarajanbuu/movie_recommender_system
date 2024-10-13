import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np

# Load datasets (adjust paths as necessary)
movies_df = pd.read_csv('C:/Users/anbum/Downloads/movies (1).csv')
ratings_df = pd.read_csv('C:/Users/anbum/Downloads/ratings (1).csv')

# Preprocessing: Encode movieId and userId
le_movie = LabelEncoder()
le_user = LabelEncoder()
ratings_df['movieId'] = le_movie.fit_transform(ratings_df['movieId'])
ratings_df['userId'] = le_user.fit_transform(ratings_df['userId'])

# Pivot table to create user-item matrix
user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Initialize KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_movie_matrix.values)

# Compute movie similarity based on genres
def compute_movie_similarity():
    # One-hot encode genres
    genre_matrix = movies_df['genres'].str.get_dummies('|')
    # Compute cosine similarity between movies
    movie_similarity = cosine_similarity(genre_matrix)
    return movie_similarity

movie_similarity = compute_movie_similarity()

def recommend_movies(user_id, genre_filter=None):
    # Find the user index
    user_index = le_user.transform([user_id])[0]
    
    # Get user's ratings from the matrix
    user_ratings = user_movie_matrix.iloc[user_index].values.reshape(1, -1)

    # Find K nearest neighbors (including the user itself)
    distances, indices = knn_model.kneighbors(user_ratings, n_neighbors=11)

    # Exclude the user's own ratings
    neighbor_indices = indices.flatten()[1:]

    # Shuffle the neighbor indices to introduce randomness
    np.random.shuffle(neighbor_indices)

    # Aggregate the neighbors' ratings and find top-rated movies not rated by the user
    neighbor_ratings = user_movie_matrix.iloc[neighbor_indices]
    avg_ratings = neighbor_ratings.mean(axis=0)
    avg_ratings = avg_ratings.sort_values(ascending=False)

    # Filter out movies already rated by the user
    user_rated_movies = user_movie_matrix.iloc[user_index][user_movie_matrix.iloc[user_index] > 0].index
    recommendations = avg_ratings.index.difference(user_rated_movies)

    # Apply genre filter if specified
    if genre_filter:
        genre_filter = genre_filter.lower()
        # Filter movies by genre
        filtered_movies = movies_df[movies_df['genres'].str.contains(genre_filter, case=False, na=False)]
        filtered_movie_ids = filtered_movies['movieId']
        recommendations = recommendations[recommendations.isin(filtered_movie_ids)]

    # Get top 10 recommended movieIds
    recommended_movie_ids = recommendations[:10]
    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)][['movieId', 'title', 'genres']]

    return recommended_movies

def recommend_similar_movies(movie_name):
    # Find the movie index
    movie_index = movies_df[movies_df['title'].str.contains(movie_name, case=False, na=False)].index
    if len(movie_index) == 0:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    
    movie_index = movie_index[0]
    # Compute similarity scores for the selected movie
    movie_similarities = movie_similarity[movie_index]
    similar_movie_indices = np.argsort(-movie_similarities)  # Sort by similarity

    # Get top 10 similar movies excluding the original movie
    similar_movies = []
    for i in similar_movie_indices:
        if i != movie_index:
            similar_movies.append(movies_df.iloc[i])
        if len(similar_movies) == 10:
            break

    return pd.DataFrame(similar_movies)[['movieId', 'title', 'genres']]

def display_genre_related_movies(user_id, genre_filter, movie_name):
    # Get movie recommendations based on the genre filter
    recommended_movies = recommend_movies(user_id, genre_filter)
    # Get similar movies based on the provided movie name
    similar_movies = recommend_similar_movies(movie_name)
    
    # Combine the results
    genre_related_movies = pd.concat([recommended_movies, similar_movies]).drop_duplicates().head(10)
    
    genre_related_movies_html = "<h2>Top 10 Genre-Related Movies:</h2><ul>"
    for _, row in genre_related_movies.iterrows():
        genre_related_movies_html += f"<li>{row['title']} ({row['genres']})</li>"
    genre_related_movies_html += "</ul>"
    display(HTML(genre_related_movies_html))

# Create widgets for user input
user_id_input = widgets.BoundedIntText(
    value=1,
    min=1,
    max=10,
    step=1,
    description='User ID:',
    disabled=False
)

genre_input = widgets.Text(
    value='',
    description='Genre Filter:',
    disabled=False
)

movie_name_input = widgets.Text(
    value='',
    description='Movie Name:',
    disabled=False
)

genre_related_button = widgets.Button(
    description='Genre-Related Top 10 Movies',
    disabled=False,
    button_style='',
    tooltip='Click to get genre-related top 10 movies',
    icon='star'
)

# Define button click event handler
def on_genre_related_button_clicked(b):
    user_id = user_id_input.value
    genre_filter = genre_input.value
    movie_name = movie_name_input.value
    display_genre_related_movies(user_id, genre_filter, movie_name)

# Attach event handler to the button
genre_related_button.on_click(on_genre_related_button_clicked)

# Display the widgets
display(user_id_input, genre_input, movie_name_input, genre_related_button)
