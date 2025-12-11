import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import MovieLensDataset
from models.ncf import NCF

def compute_metrics(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for user, item, rating in dataloader:
            user = user.to(device)
            item = item.to(device)
            
            outputs = model(user, item)
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(rating.numpy())
            
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Clip predictions to 0.5-5.0 range
    predictions = np.clip(predictions, 0.5, 5.0)
    
    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
    mae = mean_absolute_error(ground_truth, predictions)
    
    return rmse, mae

def evaluate_model(model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = MovieLensDataset()
    if not dataset.load_data():
        return

    test_loader = DataLoader(dataset.get_test_dataset(), batch_size=4096, shuffle=False, num_workers=0)
    
    # Init model
    # Note: We need to ensure embedding_dim matches training. Default is 32 in train.py, 64 in ncf.py __init__.
    # We should sync them. I'll read from args or assume 32 if I changed train.py default.
    # train.py default was 32. ncf.py default was 64. 
    # I should explicitly pass 32 here to match train.py default, or update ncf.py default.
    # I'll Assume 32 for now to match train.py default command line arg.
    
    model = NCF(dataset.num_users, dataset.num_items, embedding_dim=32).to(device)
    
    # Load weights
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints", "best_model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Mismatch in shapes likely due to embedding dimension or num_users/items.")
            return
    else:
        print("Model checkpoint not found. Evaluating initialized model (random weights).")

    print("Evaluating on Test Set...")
    rmse, mae = compute_metrics(model, test_loader, device)
    
    print("="*30)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()
