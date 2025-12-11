import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16]):
        super(MLP, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Build dense layers
        fc_layers = []
        input_size = embedding_dim * 2 # Concatenation
        
        for idx, layer_size in enumerate(layers):
            fc_layers.append(nn.Linear(input_size, layer_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.2)) # Regularization
            input_size = layer_size
            
        self.fc_layers = nn.Sequential(*fc_layers)
        self.affine_output = nn.Linear(layers[-1], 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        vector = torch.cat([user_emb, item_emb], dim=-1)
        
        # Pass through dense layers
        vector = self.fc_layers(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()
