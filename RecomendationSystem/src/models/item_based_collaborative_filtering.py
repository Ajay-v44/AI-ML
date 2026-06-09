import os
import sys
import pickle
import pandas as pd
import numpy as np

# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_ingestion.load_data import load_merged_data

class ItemBasedCollaborativeFiltering:
    """
    An Item-Based Collaborative Filtering Recommender Model.
    Uses Pearson correlation to find similar movies and predict ratings/recommendations.
    """
    def __init__(self, min_ratings=100):
        self.min_ratings = min_ratings
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.movie_stats = None

    def fit(self, df):
        """
        Trains the recommendation model by:
        1. Creating a user-item rating pivot table.
        2. Filtering out items with fewer than min_ratings.
        3. Precomputing the item-item Pearson correlation matrix.
        """
        print(f"Fitting model. Filtering movies with < {self.min_ratings} ratings...")
        
        # Calculate movie statistics (counts and average ratings)
        self.movie_stats = (
            df.groupby("title")
            .agg(
                avg_rating=("rating", "mean"),
                rating_count=("rating", "count")
            )
        )
        
        # Build pivot table
        full_pivot = df.pivot_table(
            index="user_id",
            columns="title",
            values="rating"
        )
        
        # Filter pivot columns by min_ratings to reduce noise and speed up correlation matrix computation
        popular_movies = self.movie_stats[self.movie_stats["rating_count"] >= self.min_ratings].index
        self.user_movie_matrix = full_pivot[popular_movies]
        
        print(f"User-Movie Matrix Shape after filtering: {self.user_movie_matrix.shape}")
        print("Precomputing item-item Pearson correlation matrix (this might take a few seconds)...")
        
        # Compute Pearson correlation matrix
        # min_periods ensures we have at least 10 users rating both movies to trust the correlation
        self.similarity_matrix = self.user_movie_matrix.corr(method="pearson", min_periods=10)
        
        print("Model training complete.")

    def get_similar_movies(self, movie_name, top_n=10):
        """
        Returns similar movies for a given movie.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")
            
        if movie_name not in self.similarity_matrix.columns:
            # Try a partial match if exact match is not found
            matches = [col for col in self.similarity_matrix.columns if movie_name.lower() in col.lower()]
            if matches:
                movie_name = matches[0]
                print(f"Exact match not found. Using closest match: '{movie_name}'")
            else:
                print(f"Movie '{movie_name}' not found in the trained model (or has < {self.min_ratings} ratings).")
                return pd.DataFrame()

        # Get similarities for this movie, drop NaNs, and sort
        sim_scores = self.similarity_matrix[movie_name].dropna()
        similar_movies = pd.DataFrame(sim_scores).rename(columns={movie_name: "correlation"})
        
        # Merge with stats for context
        similar_movies = similar_movies.join(self.movie_stats)
        
        # Exclude the movie itself and return top_n
        similar_movies = similar_movies[similar_movies.index != movie_name]
        return similar_movies.sort_values(by="correlation", ascending=False).head(top_n)

    def recommend(self, user_ratings, top_n=10):
        """
        Recommends items to a user based on their historical ratings.
        
        Parameters:
        -----------
        user_ratings : dict
            A dictionary mapping movie titles to user ratings.
            e.g., {"Toy Story (1995)": 5.0, "Jumanji (1995)": 3.0}
        top_n : int
            Number of recommendations to return.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing recommended titles, predicted scores, and details.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")

        # Find movies in user_ratings that are in our similarity matrix
        valid_ratings = {}
        for movie, rating in user_ratings.items():
            if movie in self.similarity_matrix.columns:
                valid_ratings[movie] = rating
            else:
                # Attempt to find closest match
                matches = [col for col in self.similarity_matrix.columns if movie.lower() in col.lower()]
                if matches:
                    valid_ratings[matches[0]] = rating

        if not valid_ratings:
            print("None of the rated movies are present in the trained model's vocabulary.")
            return pd.DataFrame()

        # Generate recommendation scores
        sim_candidates = pd.Series(dtype=float)
        sim_sums = pd.Series(dtype=float)

        for movie, rating in valid_ratings.items():
            # Get similarity series for this movie
            sim_series = self.similarity_matrix[movie].dropna()
            
            # Filter similarities (ignore weak or negative correlations)
            sim_series = sim_series[sim_series > 0.1]
            
            # Predict scores: sum(similarity * rating)
            for target_movie, similarity in sim_series.items():
                if target_movie in valid_ratings:
                    continue  # Skip movies the user has already rated
                    
                if target_movie not in sim_candidates:
                    sim_candidates[target_movie] = 0.0
                    sim_sums[target_movie] = 0.0
                
                # We weight similarity by rating
                sim_candidates[target_movie] += similarity * rating
                sim_sums[target_movie] += similarity

        if sim_candidates.empty:
            return pd.DataFrame()

        # Divide by sum of similarities to normalize
        rec_scores = sim_candidates / (sim_sums + 1e-9)
        
        # Create results DataFrame
        recommendations = pd.DataFrame(rec_scores, columns=["score"])
        recommendations = recommendations.join(self.movie_stats)
        
        return recommendations.sort_values(by="score", ascending=False).head(top_n)

    def save(self, filepath):
        """
        Saves the trained model to a pickle file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Loads a trained model from a pickle file.
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"Model successfully loaded from: {filepath}")
        return model

if __name__ == "__main__":
    # 1. Load data
    print("Loading data...")
    df = load_merged_data()
    
    # 2. Train Model
    model = ItemBasedCollaborativeFiltering(min_ratings=100)
    model.fit(df)
    
    # 3. Test similar movie recommendation
    test_movie = "Toy Story (1995)"
    print(f"\nTesting: Similar movies to '{test_movie}':")
    similar_df = model.get_similar_movies(test_movie, top_n=5)
    print(similar_df[["correlation", "avg_rating", "rating_count"]])
    
    # 4. Test user recommendation
    sample_user_ratings = {
        "Toy Story (1995)": 5.0,
        "Aladdin (1992)": 4.5,
        "Jumanji (1995)": 2.0
    }
    print(f"\nTesting: User recommendations based on ratings {sample_user_ratings}:")
    rec_df = model.recommend(sample_user_ratings, top_n=5)
    print(rec_df[["score", "avg_rating", "rating_count"]])
    
    # 5. Save Model
    model_save_path = os.path.abspath(os.path.join(PROJECT_ROOT, "data/processed/item_based_cf_model.pkl"))
    model.save(model_save_path)
