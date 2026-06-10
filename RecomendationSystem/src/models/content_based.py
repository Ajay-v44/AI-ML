import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/processed/imdb_movies.csv"
    )
)

movies = pd.read_csv(data_path)

print(movies.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    movies["features"]
)

print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

indices = pd.Series(
    movies.index,
    index=movies["primaryTitle"]
).drop_duplicates()

def recommend(movie_title, top_n=10):

    idx = indices[movie_title]

    similarity_scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:top_n+1
    ]

    movie_indices = [
        score[0]
        for score in similarity_scores
    ]

    return movies[
        [
            "primaryTitle",
            "genres",
            "averageRating"
        ]
    ].iloc[movie_indices]

print(
    recommend(
        "Inception"
    )
)