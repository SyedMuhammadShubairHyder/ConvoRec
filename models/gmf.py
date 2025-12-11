import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Output layer
        self.affine_output = nn.Linear(embedding_dim, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Element-wise product
        element_product = torch.mul(user_emb, item_emb)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating.squeeze()
