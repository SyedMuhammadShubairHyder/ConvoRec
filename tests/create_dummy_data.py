import pandas as pd
import numpy as np
import os

def create_dummy_data():
    # Create a small dataframe
    data = {
        'userId': np.random.randint(0, 100, 1000),
        'movieId': np.random.randint(0, 50, 1000),
        'rating': np.random.randint(1, 6, 1000), # 1-5
        'timestamp': np.random.randint(100000, 200000, 1000)
    }
    df = pd.DataFrame(data)
    
    # Split
    train = df.iloc[:800]
    val = df.iloc[800:900]
    test = df.iloc[900:]
    
    os.makedirs("data/processed", exist_ok=True)
    train.to_pickle("data/processed/train.pkl")
    val.to_pickle("data/processed/val.pkl")
    test.to_pickle("data/processed/test.pkl")
    print("Dummy data created in data/processed/")

if __name__ == "__main__":
    create_dummy_data()
