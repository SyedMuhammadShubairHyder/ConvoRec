import pandas as pd
import numpy as np
import os
import pickle

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "ml-25m")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def preprocess():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print(f"Created directory: {PROCESSED_DIR}")
    
    ratings_path = os.path.join(RAW_DIR, "ratings.csv")
    movies_path = os.path.join(RAW_DIR, "movies.csv")
    
    if not os.path.exists(ratings_path):
        # Fallback if raw dir structure is different
        RAW_DIR_FALLBACK = os.path.join(BASE_DIR, "data", "raw")
        ratings_path = os.path.join(RAW_DIR_FALLBACK, "ratings.csv")
        movies_path = os.path.join(RAW_DIR_FALLBACK, "movies.csv")
        if not os.path.exists(ratings_path):
            print(f"Error: Could not find ratings.csv in {RAW_DIR} or {RAW_DIR_FALLBACK}")
            return

    print(f"Loading data from {ratings_path}...")
    # Load only necessary columns to save memory
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'timestamp'])
    movies = pd.read_csv(movies_path)
    
    print(f"Original ratings: {len(ratings)}")
    
    # Filter users with < 20 ratings
    print("Filtering users...")
    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index
    ratings = ratings[ratings['userId'].isin(active_users)]
    print(f"Filtered ratings (>=20/user): {len(ratings)}")
    
    # Convert timestamp to datetime
    print("Converting timestamps...")
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Split based on time
    # Train: < 2018
    # Val: 2018-2019
    # Test: >= 2019
    
    val_start = pd.Timestamp('2018-01-01')
    test_start = pd.Timestamp('2019-01-01')
    
    print("Splitting data...")
    train_data = ratings[ratings['timestamp'] < val_start].copy()
    val_data = ratings[(ratings['timestamp'] >= val_start) & (ratings['timestamp'] < test_start)].copy()
    test_data = ratings[ratings['timestamp'] >= test_start].copy()
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Save
    print("Saving processed files...")
    train_data.to_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    val_data.to_pickle(os.path.join(PROCESSED_DIR, "val.pkl"))
    test_data.to_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    movies.to_pickle(os.path.join(PROCESSED_DIR, "movies.pkl"))
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
