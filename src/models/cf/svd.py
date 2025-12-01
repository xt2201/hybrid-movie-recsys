import numpy as np
import scipy.sparse as sparse
import yaml
import pickle
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yml"

class SVDRecommender:
    """
    SVD-based collaborative filtering using Alternating Least Squares (ALS).
    Better suited for implicit feedback / binary interaction data.
    """
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.cf_config = self.config["model"]["cf"]
        self.factors = self.cf_config["factors"]
        self.regularization = self.cf_config.get("regularization", 0.01)
        self.iterations = self.cf_config.get("iterations", 20)
        self.alpha = self.cf_config.get("alpha", 40)  # Confidence weight for implicit feedback
        
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix: sparse.csr_matrix):
        """
        Train using Weighted Alternating Least Squares (ALS) for implicit feedback.
        Reference: "Collaborative Filtering for Implicit Feedback Datasets" (Hu et al., 2008)
        """
        logger.info("Training SVD/ALS model...")
        
        # Convert to implicit feedback: C = 1 + alpha * R (confidence)
        # P = 1 if R > 0 else 0 (preference)
        Cui = user_item_matrix.copy()
        Cui.data = 1.0 + self.alpha * Cui.data  # Confidence
        
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.01, (n_items, self.factors))
        
        # Regularization matrix
        reg_I = self.regularization * np.eye(self.factors)
        
        # Convert to csc for efficient column access
        Cui_csc = Cui.tocsc()
        Cui_csr = Cui.tocsr()
        
        for iteration in range(self.iterations):
            # Update user factors
            self._als_step(Cui_csr, self.user_factors, self.item_factors, reg_I)
            # Update item factors
            self._als_step(Cui_csc.T.tocsr(), self.item_factors, self.user_factors, reg_I)
            
            if (iteration + 1) % 5 == 0:
                logger.info(f"  Iteration {iteration + 1}/{self.iterations}")
        
        logger.info("Training complete.")

    def _als_step(self, Cui, X, Y, reg_I):
        """One ALS step: update X given Y fixed."""
        YtY = Y.T @ Y
        
        for u in range(X.shape[0]):
            # Get non-zero entries for this user/item
            start = Cui.indptr[u]
            end = Cui.indptr[u + 1]
            
            indices = Cui.indices[start:end]
            confidence = Cui.data[start:end]
            
            # A = Y^T C Y + reg*I
            # For efficiency, A = Y^T Y + Y^T (C-I) Y + reg*I
            # where (C-I) is diagonal with only non-zero entries
            
            if len(indices) > 0:
                Y_u = Y[indices]  # Items this user interacted with
                Cu_diag = confidence - 1  # C_u - I (only non-zero items)
                
                A = YtY + Y_u.T @ (Cu_diag.reshape(-1, 1) * Y_u) + reg_I
                
                # b = Y^T C p, where p = 1 for interacted items
                b = Y_u.T @ confidence  # Since p=1 for all non-zero
                
                X[u] = np.linalg.solve(A, b)
            else:
                # No interactions, just regularization
                X[u] = np.zeros(self.factors)

    def recommend(self, user_id: int, user_item_matrix: sparse.csr_matrix, N: int = 10):
        """Recommend top-N items for a user."""
        if self.user_factors is None:
            raise RuntimeError("Model not fitted")
        
        # Predict scores for all items
        user_vector = self.user_factors[user_id]
        scores = self.item_factors @ user_vector
        
        # Mask already interacted items
        user_row = user_item_matrix[user_id]
        interacted_indices = user_row.indices
        scores[interacted_indices] = -np.inf
        
        # Get top N
        top_indices = np.argsort(scores)[-N:][::-1]
        top_scores = scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'factors': self.factors
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.user_factors = data['user_factors']
            self.item_factors = data['item_factors']
            self.factors = data['factors']
        logger.info(f"Model loaded from {path}")
