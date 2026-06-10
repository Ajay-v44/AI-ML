import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

basics_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/raw/title.basics.tsv"
    )
)

ratings_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/raw/title.ratings.tsv"
    )
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

print("Loading IMDb Basics...")

basics = pd.read_csv(
    basics_path,
    sep="\t",
    low_memory=False
)

print("Loading IMDb Ratings...")

ratings = pd.read_csv(
    ratings_path,
    sep="\t"
)

# --------------------------------------------------
# Basic Validation
# --------------------------------------------------

print("\nBasics Shape:")
print(basics.shape)

print("\nRatings Shape:")
print(ratings.shape)

print("\nBasics Columns:")
print(basics.columns)

print("\nRatings Columns:")
print(ratings.columns)

movies = basics[
    basics["titleType"] == "movie"
]

print("\nMovies Only:")
print(movies.shape)

movies = movies.replace(
    "\\N",
    pd.NA
)


movies = movies.dropna(
    subset=[
        "genres",
        "startYear"
    ]
)


imdb_df = movies.merge(
    ratings,
    on="tconst",
    how="inner"
)

print("\nMerged Shape:")
print(imdb_df.shape)


imdb_df = imdb_df[
    [
        "tconst",
        "primaryTitle",
        "startYear",
        "runtimeMinutes",
        "genres",
        "averageRating",
        "numVotes"
    ]
]

print(imdb_df.head())


imdb_df = imdb_df[
    imdb_df["numVotes"] >= 5000
]
imdb_df = imdb_df.dropna(
    subset=[
        "runtimeMinutes",
        "genres"
    ]
)

imdb_df["startYear"] = imdb_df[
    "startYear"
].astype(int)

imdb_df["runtimeMinutes"] = imdb_df[
    "runtimeMinutes"
].astype(int)

print(imdb_df.shape, "shape after filtering")

imdb_df["features"] = (
    imdb_df["genres"].astype(str)
    + " "
    + imdb_df["startYear"].astype(str)
)

output_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/processed/imdb_movies.csv"
    )
)

imdb_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)

