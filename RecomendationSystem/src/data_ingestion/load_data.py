import os
import pandas as pd

# Resolve absolute paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_path = os.path.abspath(os.path.join(BASE_DIR, "../../data/raw/movies.dat"))
ratings_path = os.path.abspath(os.path.join(BASE_DIR, "../../data/raw/ratings.dat"))
users_path = os.path.abspath(os.path.join(BASE_DIR, "../../data/raw/users.dat"))

def load_raw_data():
    """
    Loads raw MovieLens datasets.
    """
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        encoding="latin1",
        names=[
            "movie_id",
            "title",
            "genres"
        ]
    )
    
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )
    
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "gender",
            "age",
            "occupation",
            "zip_code"
        ]
    )
    
    return movies, ratings, users

def load_merged_data():
    """
    Loads raw datasets and merges ratings with movie details.
    """
    movies, ratings, users = load_raw_data()
    df = ratings.merge(movies, on="movie_id")
    return df

if __name__ == "__main__":
    movies, ratings, users = load_raw_data()
    
    print(movies.head())
    print(ratings.head())
    print(users.head())

    print("Shapes:")
    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)
    print("Users:", users.shape)

    print("Unique users:", ratings["user_id"].nunique())
    print("Unique movies:", ratings["movie_id"].nunique())

    df = load_merged_data()
    print(df.head())

    movie_stats = (
        df.groupby("title")
          .agg(
              avg_rating=("rating", "mean"),
              rating_count=("rating", "count")
          )
    )

    movie_stats = movie_stats[
        movie_stats["rating_count"] > 100
    ]

    top_movies = movie_stats.sort_values(
        "avg_rating",
        ascending=False
    )

    print("\nTop 20 Movies by Average Rating (with > 100 ratings):")
    print(top_movies.head(20))

    print("===========")

    # Collaborative Filtering Sparsity Analysis
    user_movie_matrix = df.pivot_table(
        index="user_id",
        columns="title",
        values="rating"
    )

    print("User-Movie Matrix Shape:", user_movie_matrix.shape)

    total_cells = (
        user_movie_matrix.shape[0]
        * user_movie_matrix.shape[1]
    )

    filled_cells = (
        user_movie_matrix.notna()
        .sum()
        .sum()
    )

    sparsity = (1 - (filled_cells / total_cells)) * 100
    print(f"Sparsity: {sparsity:.2f}%")

    def get_recommedation(movie_name):
        movie_ratings = user_movie_matrix[movie_name]
        similar_movies = user_movie_matrix.corrwith(movie_ratings)
        corr_df = pd.DataFrame(similar_movies, columns=["correlation"])
        corr_df.dropna(inplace=True)

        rating_counts = (
            df.groupby("title")
            .size()
            .reset_index(name="rating_count")
        )

        corr_df = corr_df.merge(
            rating_counts,
            left_index=True,
            right_on="title"
        )
        corr_df = corr_df[
            corr_df["rating_count"] > 100
        ]

        return corr_df.sort_values(
            "correlation",
            ascending=False
        )

    movie_name = "Toy Story (1995)"
    print(f"\nMovies similar to {movie_name}:")
    print(
        get_recommedation(movie_name)[
            ["title", "correlation"]
        ].head(20)
    )

    movie_name = "Jumanji (1995)"
    print(f"\nMovies similar to {movie_name}:")
    print(
        get_recommedation(movie_name)[
            ["title", "correlation"]
        ].head(20)
    )


