import argparse
import os
import sys

from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

# Add project root to sys.path so we can import project modules.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_ingestion.load_data import load_raw_data
from src.models.svd import SVDRecommender


def load_surprise_dataset() -> Dataset:
    """
    Load ratings and convert them into Surprise's internal dataset format.
    """
    _, ratings_df, _ = load_raw_data()
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]],
        reader,
    )


def find_best_svd_params() -> dict:
    """
    Run a simple grid search to find better SVD hyperparameters.

    This is a form of fine-tuning because we are searching for parameter values
    that reduce prediction error on validation folds.
    """
    data = load_surprise_dataset()

    # Start with a small search space so it is easier to understand and faster to run.
    param_grid = {
        "n_factors": [50, 100, 150],
        "n_epochs": [20, 30],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.02, 0.1],
    }

    grid_search = GridSearchCV(
        algo_class=SVD,
        param_grid=param_grid,
        measures=["rmse", "mae"],
        cv=3,
        joblib_verbose=1,
    )

    print("Running grid search for SVD hyperparameters...")
    grid_search.fit(data)

    print("\nBest RMSE score:", grid_search.best_score["rmse"])
    print("Best RMSE params:", grid_search.best_params["rmse"])
    print("Best MAE score:", grid_search.best_score["mae"])
    print("Best MAE params:", grid_search.best_params["mae"])

    return grid_search.best_params["rmse"]


def train_best_model(save_path: str) -> None:
    """
    1. Search for better hyperparameters.
    2. Retrain on the full dataset with the best values.
    3. Save the final model.
    """
    best_params = find_best_svd_params()

    _, ratings_df, _ = load_raw_data()

    recommender = SVDRecommender(
        n_factors=best_params["n_factors"],
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
    )

    # Train on the full dataset after tuning so the model uses all available ratings.
    recommender.fit_full_trainset(ratings_df)
    recommender.save(save_path)

    print("\nFine-tuned model training complete.")
    print(f"Saved fine-tuned model to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Surprise SVD recommender with grid search.",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join(PROJECT_ROOT, "data/processed/svd_finetuned_model.pkl"),
        help="Where to save the fine-tuned model.",
    )
    args = parser.parse_args()

    train_best_model(args.save_path)


if __name__ == "__main__":
    main()
