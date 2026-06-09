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


