import pandas as pd
import os
import numpy as np

class MovieLensDataset:
    def __init__(self, data_path="data/raw/ml-25m", min_ratings=20):
        self.data_path = data_path
        self.min_ratings = min_ratings
        
        self.ratings_path = os.path.join(data_path, "ratings.csv")
        self.movies_path = os.path.join(data_path, "movies.csv")
        
        self.users = None
        self.items = None
        self.ratings = None
        self.train = None
        self.val = None
        self.test = None

    def load_data(self):
        """Loads ratings and movies data."""
        print("Loading ratings...")
        self.ratings = pd.read_csv(self.ratings_path)
        print(f"Loaded {len(self.ratings)} ratings.")
        
        print("Loading movies...")
        self.movies = pd.read_csv(self.movies_path)
        print(f"Loaded {len(self.movies)} movies.")
        
        # Filter sparse users/items if needed
        if self.min_ratings > 0:
            self._filter_sparse_data()
            
    def _filter_sparse_data(self):
        print(f"Filtering users with fewer than {self.min_ratings} ratings...")
        user_counts = self.ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= self.min_ratings].index
        self.ratings = self.ratings[self.ratings['userId'].isin(active_users)].copy()
        print(f"Remaining ratings: {len(self.ratings)}")

    def split_data(self, val_ratio=0.1, test_ratio=0.1):
        """Splits data into train, val, test using leave-one-out or temporal split.
        For simplicity here, we use a random split or time-based if timestamp is available.
        Using random split for now to ensure coverage.
        """
        print("Splitting data...")
        # Sort by timestamp to make it physically realistic (past -> future)
        self.ratings = self.ratings.sort_values('timestamp')
        
        n = len(self.ratings)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - test_size - val_size
        
        self.train = self.ratings.iloc[:train_size]
        self.val = self.ratings.iloc[train_size:train_size + val_size]
        self.test = self.ratings.iloc[train_size + val_size:]
        
        print(f"Train: {len(self.train)}, Val: {len(self.val)}, Test: {len(self.test)}")

    def save_processed(self, output_dir="data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        self.train.to_pickle(os.path.join(output_dir, "train.pkl"))
        self.val.to_pickle(os.path.join(output_dir, "val.pkl"))
        self.test.to_pickle(os.path.join(output_dir, "test.pkl"))
        print(f"Saved processed data to {output_dir}")

if __name__ == "__main__":
    # Test the loader
    dataset = MovieLensDataset()
    if os.path.exists(dataset.ratings_path):
        dataset.load_data()
        dataset.split_data()
        dataset.save_processed()
    else:
        print("Data not found. Run scripts/download_data.py first.")
