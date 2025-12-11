import torch
import torch.nn as nn
from .gmf import GMF
from .mlp import MLP

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_emb_dim=32, mlp_emb_dim=32, mlp_layers=[64, 32, 16]):
        """
        Neural Matrix Factorization (NeuMF) combining GMF and MLP.
        """
        super(NeuMF, self).__init__()
        
        # GMF Part
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_emb_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_emb_dim)
        
        # MLP Part
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_emb_dim)
        
        fc_layers = []
        input_size = mlp_emb_dim * 2
        for layer_size in mlp_layers:
            fc_layers.append(nn.Linear(input_size, layer_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2))
            input_size = layer_size
        self.mlp_layers = nn.Sequential(*fc_layers)
        
        # Final prediction layer
        # Input is concatenation of GMF output (element product, size=gmf_emb_dim) 
        # and MLP output (last layer size)
        predict_input_size = gmf_emb_dim + mlp_layers[-1]
        self.affine_output = nn.Linear(predict_input_size, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # GMF Forward
        gmf_user = self.gmf_user_embedding(user_indices)
        gmf_item = self.gmf_item_embedding(item_indices)
        gmf_vector = torch.mul(gmf_user, gmf_item)
        
        # MLP Forward
        mlp_user = self.mlp_user_embedding(user_indices)
        mlp_item = self.mlp_item_embedding(item_indices)
        mlp_vector = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        # Concatenate and Predict
        final_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logits = self.affine_output(final_vector)
        rating = self.logistic(logits)
        return rating.squeeze()
