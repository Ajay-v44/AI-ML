import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
ratings_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../data/raw/ratings.dat"
    )
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

print(ratings.head())


from surprise import Dataset
from surprise import Reader

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[
        [
            "user_id",
            "movie_id",
            "rating"
        ]
    ],
    reader
)

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

from surprise import SVD

model = SVD()

model.fit(trainset)

predictions = model.test(
    testset
)
from surprise import accuracy

accuracy.rmse(
    predictions
)

prediction = model.predict(
    uid=1,
    iid=50
)

print(prediction.est)

import pickle

with open(
    os.path.join(
        PROJECT_ROOT,
        "data/processed/svd_model.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )