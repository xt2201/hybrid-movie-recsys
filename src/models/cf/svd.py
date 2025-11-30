import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import TruncatedSVD
import yaml
import pickle
import os

CONFIG_PATH = "config/config.yml"

class SVDRecommender:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.cf_config = self.config["model"]["cf"]
        self.factors = self.cf_config["factors"]
        # regularization and iterations are not directly used in TruncatedSVD (it uses randomized SVD)
        
        self.model = TruncatedSVD(n_components=self.factors, random_state=42)
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix: sparse.csr_matrix):
        print("Training SVD model...")
        # TruncatedSVD expects (n_samples, n_features). 
        # If we treat users as samples and items as features, we get user_factors (U*Sigma) and item_factors (V^T).
        self.user_factors = self.model.fit_transform(user_item_matrix)
        self.item_factors = self.model.components_
        print("Training complete.")

    def recommend(self, user_id: int, user_item_matrix: sparse.csr_matrix, N: int = 10):
        # Predict scores for all items
        # score = user_factor * item_factors
        if self.user_factors is None:
            raise RuntimeError("Model not fitted")
            
        user_vector = self.user_factors[user_id]
        scores = np.dot(user_vector, self.item_factors)
        
        # Mask already interacted items
        # user_item_matrix is csr, so get indices
        user_row = user_item_matrix[user_id]
        interacted_indices = user_row.indices
        scores[interacted_indices] = -np.inf
        
        # Get top N
        top_indices = scores.argsort()[-(N):][::-1]
        top_scores = scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
