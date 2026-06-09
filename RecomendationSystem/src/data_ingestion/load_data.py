from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# The HF_TOKEN environment variable will be automatically picked up by load_dataset
dataset = load_dataset("reczoo/Movielens1M_m1")

print(dataset)

# The dataset has 'train', 'validation', and 'test' splits, not 'ratings' and 'movies'
train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()

print(train_df.head())
print(val_df.head())


train_df.to_csv(
    "../../data/raw/train.csv",
    index=False
)

val_df.to_csv(
    "../../data/raw/validation.csv",
    index=False
)


