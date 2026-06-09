import pandas as pd

ratings = pd.read_csv("../raw/ratings.csv")
movies = pd.read_csv("../raw/movies.csv")

print("Ratings Shape:", ratings.shape)
print("Movies Shape:", movies.shape)


print(
    "Users:",
    ratings["userId"].nunique()
)


print(
    "Movies:",
    ratings["movieId"].nunique()
)


import matplotlib.pyplot as plt

ratings["rating"].hist(
    bins=10
)

plt.title("Rating Distribution")
plt.show()


top_movies = (
    ratings.groupby("movieId")
    .size()
    .sort_values(
        ascending=False
    )
    .head(10)
)

print(top_movies)


movie_data = ratings.merge(
    movies,
    on="movieId"
)

print(movie_data.head())


avg_rating = (
    movie_data.groupby("title")
    ["rating"]
    .mean()
)

rating_count = (
    movie_data.groupby("title")
    ["rating"]
    .count()
)


movie_stats = pd.DataFrame({
    "avg_rating": avg_rating,
    "rating_count": rating_count
})



movie_stats = movie_stats[
    movie_stats["rating_count"] > 50
]


top_movies = (
    movie_stats
    .sort_values(
        "avg_rating",
        ascending=False
    )
)

print(top_movies.head(20))


movies["genres"] = (
    movies["genres"]
    .fillna("")
)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["genres"]
)


print(tfidf_matrix.shape)



from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(
    tfidf_matrix
)


indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


def recommend(movie_title):

    idx = indices[movie_title]

    scores = list(
        enumerate(similarity[idx])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:11]

    movie_indices = [
        i[0]
        for i in scores
    ]

    return movies[
        "title"
    ].iloc[
        movie_indices
    ]


print(
    recommend(
        "Toy Story (1995)"
    )
)


