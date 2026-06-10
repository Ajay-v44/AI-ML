# User-Based Collaborative Filtering

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/raw/movies.dat"
    )
)

ratings_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/raw/ratings.dat"
    )
)

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

df = ratings.merge(
    movies,
    on="movie_id"
)

user_movie_matrix = df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

print(user_movie_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(
    user_movie_matrix
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def get_similar_users(
    user_id,
    top_n=10
):

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(
            ascending=False
        )
    )

    return similar_users.iloc[
        1:top_n+1
    ]
print("Similar Users:")
print(
    get_similar_users(1)
)


def get_watched_movies(
    user_id
):

    watched = ratings[
        ratings["user_id"] == user_id
    ]

    return set(
        watched["movie_id"]
    )

def recommend_movies(
    user_id,
    top_n=10
):

    similar_users = (
        get_similar_users(
            user_id
        ).index
    )

    watched_movies = (
        get_watched_movies(
            user_id
        )
    )

    recommendations = []

    for sim_user in similar_users:

        sim_ratings = ratings[
            ratings["user_id"]
            == sim_user
        ]

        sim_ratings = sim_ratings[
            sim_ratings["rating"] >= 4
        ]

        recommendations.append(
            sim_ratings
        )

    recommendations = pd.concat(
        recommendations
    )

    recommendations = recommendations[
        ~recommendations[
            "movie_id"
        ].isin(
            watched_movies
        )
    ]

    movie_scores = (
        recommendations
        .groupby("movie_id")
        ["rating"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    recommended_movies = (
        movies[
            movies["movie_id"]
            .isin(
                movie_scores.head(top_n).index
            )
        ]
    )

    return recommended_movies

