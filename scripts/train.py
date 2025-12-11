import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ncf import NeuMF
from models.gmf import GMF
from models.mlp import MLP

class RatingDataset(Dataset):
    def __init__(self, data_path):
        if data_path.endswith('.pkl'):
            self.df = pd.read_pickle(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        # Ensure user/item IDs are 0-indexed for embedding layers
        # Note: In a real scenario, we need a consistent mapping from raw IDs to indices across train/val/test
        # For this example, we assume indices are already handled or we re-map quickly (simplification)
        self.user_tensor = torch.LongTensor(self.df['userId'].values.astype(np.int64))
        self.item_tensor = torch.LongTensor(self.df['movieId'].values.astype(np.int64))
        
        # Normalize ratings to 0-1 for Sigmoid output (if using BCE) or keep raw for MSE
        # Here we normalize 0.5-5.0 to 0.0-1.0
        self.ratings = torch.FloatTensor(self.df['rating'].values / 5.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx], self.ratings[idx]

def train(model_name="NeuMF", epochs=5, batch_size=256, lr=0.001):
    # Paths
    train_path = "data/processed/train.pkl"
    val_path = "data/processed/val.pkl"
    
    if not os.path.exists(train_path):
        print("Processed data not found. Please run data_loader.py first.")
        return

    print(f"Loading data for {model_name}...")
    train_dataset = RatingDataset(train_path)
    val_dataset = RatingDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine max user/item ID for embeddings
    # (In production, use a shared mapping. Here we take max of train set roughly)
    num_users = train_dataset.df['userId'].max() + 1
    num_items = train_dataset.df['movieId'].max() + 1
    
    print(f"Num Users: {num_users}, Num Items: {num_items}")

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_name == "GMF":
        model = GMF(num_users, num_items).to(device)
    elif model_name == "MLP":
        model = MLP(num_users, num_items).to(device)
    else:
        model = NeuMF(num_users, num_items).to(device)

    criterion = nn.BCELoss() # Binary Cross Entropy because we normalized ratings 0-1 and use Sigmoid
    # Alternatively use MSELoss for regression without sigmoid
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for user, item, rating in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user, item, rating in val_loader:
                user, item, rating = user.to(device), item.to(device), rating.to(device)
                prediction = model(user, item)
                loss = criterion(prediction, rating)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
    print("Training complete.")
    torch.save(model.state_dict(), f"models/{model_name}_best.pth")
    print(f"Model saved to models/{model_name}_best.pth")

if __name__ == "__main__":
    train() # Defaults to NeuMF
