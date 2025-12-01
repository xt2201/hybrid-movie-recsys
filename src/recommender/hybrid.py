import numpy as np
import scipy.sparse as sparse
import yaml
from typing import List, Tuple
from src.models.cf.svd import SVDRecommender
from src.models.content.tfidf import ContentRecommender
from src.data.dataset import MovieDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yml"

class HybridRecommender:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.hybrid_config = self.config["model"]["hybrid"]
        self.alpha_cf = self.hybrid_config["alpha_cf"]
        self.alpha_content = self.hybrid_config["alpha_content"]
        
        self.cf_model = SVDRecommender(config_path)
        self.content_model = ContentRecommender(config_path)
        self.dataset = MovieDataset(config_path)
        
        self.is_fitted = False

    def fit(self):
        logger.info("Fitting Hybrid Recommender...")
        self.dataset.load_data()
        
        # Fit CF
        interaction_matrix = self.dataset.get_interaction_matrix()
        self.cf_model.fit(interaction_matrix)
        
        # Fit Content
        content_df = self.dataset.get_content_features()
        self.content_model.fit(content_df)
        
        self.is_fitted = True
        logger.info("Hybrid Recommender fitted.")

    def recommend(self, user_id: int, N: int = 10, user_item_matrix: sparse.csr_matrix = None) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        # Use passed matrix or internal one
        if user_item_matrix is None:
            user_item_matrix = self.dataset.get_interaction_matrix()

        # 1. Get CF candidates
        # We ask for more candidates to re-rank
        cf_ids, cf_scores = self.cf_model.recommend(user_id, user_item_matrix, N=N*2)
        
        # Normalize CF scores (min-max or just max)
        if len(cf_scores) > 0:
            cf_scores = np.array(cf_scores)
            cf_scores = cf_scores / np.max(cf_scores)
        
        cf_dict = dict(zip(cf_ids, cf_scores))
        
        # 2. Get Content candidates
        # For content, we need a reference item. 
        # Strategy: Get user's last liked item or top rated item.
        # For simplicity, let's take the item with highest rating from user history.
        
        # user_id passed here is assumed to be internal index
        
        # Get user history
        user_row = user_item_matrix[user_id]
        user_items = user_row.indices
        user_ratings = user_row.data
        
        final_scores = {}
        
        if len(user_items) > 0:
            # Find best rated item
            best_item_idx = user_items[np.argmax(user_ratings)]
            
            # Get similar items
            content_ids, content_scores = self.content_model.recommend(best_item_idx, N=N*2)
            
            if len(content_scores) > 0:
                content_scores = np.array(content_scores)
                # content scores are cosine similarity (0-1), so already normalized-ish
            
            content_dict = dict(zip(content_ids, content_scores))
        else:
            content_dict = {}

        # 3. Combine
        all_candidates = set(cf_ids) | set(content_dict.keys())
        
        for item_idx in all_candidates:
            s_cf = cf_dict.get(item_idx, 0.0)
            s_content = content_dict.get(item_idx, 0.0)
            
            score = self.alpha_cf * s_cf + self.alpha_content * s_content
            final_scores[item_idx] = score
            
        # Sort
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:N]

if __name__ == "__main__":
    recsys = HybridRecommender()
    recsys.fit()
    
    user_id = 0
    recs = recsys.recommend(user_id)
    logger.info(f"Hybrid recommendations for user {user_id}: {recs}")
