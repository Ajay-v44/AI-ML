import os
import sys
import pickle
from typing import Dict, Optional

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Add project root to sys.path so this file can be run directly as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_ingestion.load_data import load_raw_data


class SVDRecommender:
    """
    A small wrapper around Surprise's SVD model.

    Why this class is useful:
    - It keeps the training logic in one place.
    - It gives us easy-to-read methods like fit(), predict_rating(), and save().
    - It is easier to fine-tune later because all model parameters are stored together.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ) -> None:
        # These values are the main SVD hyperparameters we usually tune.
        self.params = {
            "n_factors": n_factors,
            "n_epochs": n_epochs,
            "lr_all": lr_all,
            "reg_all": reg_all,
            "random_state": random_state,
        }
        self.model = SVD(**self.params)
        self.trainset = None
        self.testset = None
        self.metrics: Dict[str, float] = {}

    def _build_surprise_dataset(self, ratings_df: pd.DataFrame) -> Dataset:
        """
        Convert a pandas DataFrame into the Surprise dataset format.

        Surprise expects exactly: user, item, rating.
        """
        reader = Reader(rating_scale=(1, 5))
        return Dataset.load_from_df(
            ratings_df[["user_id", "movie_id", "rating"]],
            reader,
        )

    def fit(
        self,
        ratings_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the SVD model and evaluate it on a holdout test split.

        Returns a dictionary with evaluation metrics so it is easy to inspect
        model quality after training.
        """
        data = self._build_surprise_dataset(ratings_df)

        # Split the data into train and test portions.
        self.trainset, self.testset = train_test_split(
            data,
            test_size=test_size,
            random_state=self.params["random_state"],
        )

        print("Training SVD model...")
        self.model.fit(self.trainset)

        print("Evaluating SVD model...")
        predictions = self.model.test(self.testset)

        # RMSE and MAE are common metrics for rating prediction tasks.
        rmse_value = accuracy.rmse(predictions, verbose=True)
        mae_value = accuracy.mae(predictions, verbose=True)

        self.metrics = {
            "rmse": rmse_value,
            "mae": mae_value,
        }
        return self.metrics

    def fit_full_trainset(self, ratings_df: pd.DataFrame) -> None:
        """
        Train the model on the full dataset.

        Use this after you decide on the best hyperparameters, because now
        we want the model to learn from all available ratings.
        """
        data = self._build_surprise_dataset(ratings_df)
        full_trainset = data.build_full_trainset()
        print("Training SVD model on the full dataset...")
        self.model.fit(full_trainset)
        self.trainset = full_trainset
        self.testset = None

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user may give to a movie.
        """
        prediction = self.model.predict(uid=user_id, iid=movie_id)
        return prediction.est

    def recommend_for_user(
        self,
        user_id: int,
        movies_df: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend top movies for a given user.

        How this works:
        - Surprise creates an "anti-testset", which means all movies the user
          has not rated yet.
        - We predict ratings for those unseen movies.
        - Then we sort by the estimated rating.
        """
        if self.trainset is None:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")

        anti_testset = self.trainset.build_anti_testset()
        user_predictions = self.model.test(
            [row for row in anti_testset if row[0] == user_id]
        )

        recommendations = [
            {"movie_id": int(pred.iid), "predicted_rating": pred.est}
            for pred in user_predictions
        ]

        recommendations_df = pd.DataFrame(recommendations)
        if recommendations_df.empty:
            return recommendations_df

        recommendations_df = recommendations_df.merge(
            movies_df[["movie_id", "title", "genres"]],
            on="movie_id",
            how="left",
        )

        return recommendations_df.sort_values(
            by="predicted_rating",
            ascending=False,
        ).head(top_n)

    def save(self, filepath: str) -> None:
        """
        Save the trained recommender so we can reuse it later without retraining.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SVDRecommender":
        """
        Load a previously saved recommender.
        """
        with open(filepath, "rb") as file:
            return pickle.load(file)


def load_ratings_data() -> pd.DataFrame:
    """
    Load only the ratings table because SVD learns from user-item-rating data.
    """
    _, ratings_df, _ = load_raw_data()
    return ratings_df


def train_svd_model(
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    test_size: float = 0.2,
    model_path: Optional[str] = None,
) -> SVDRecommender:
    """
    Train an SVD model with configurable hyperparameters.

    This is the main function you can import and use from other files.
    """
    ratings_df = load_ratings_data()

    recommender = SVDRecommender(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
    )
    metrics = recommender.fit(ratings_df, test_size=test_size)

    print(f"Finished training. Metrics: {metrics}")

    if model_path:
        recommender.save(model_path)

    return recommender


def predict_with_saved_model(
    model_path: str,
    user_id: int,
    movie_id: int,
) -> float:
    """
    Load a saved model and predict one user-movie rating.
    """
    recommender = SVDRecommender.load(model_path)
    return recommender.predict_rating(user_id=user_id, movie_id=movie_id)


if __name__ == "__main__":
    ratings_df = load_ratings_data()
    movies_df, _, _ = load_raw_data()

    save_path = os.path.join(PROJECT_ROOT, "data/processed/svd_model.pkl")

    # Train a default SVD model.
    recommender = SVDRecommender()
    recommender.fit(ratings_df)
    recommender.save(save_path)

    # Predict one sample rating.
    predicted_rating = recommender.predict_rating(user_id=1, movie_id=50)
    print(f"Predicted rating for user 1 and movie 50: {predicted_rating:.3f}")

    # Show a few recommendations for learning/demo purposes.
    top_recommendations = recommender.recommend_for_user(
        user_id=1,
        movies_df=movies_df,
        top_n=5,
    )
    print("\nTop recommendations for user 1:")
    print(top_recommendations)
