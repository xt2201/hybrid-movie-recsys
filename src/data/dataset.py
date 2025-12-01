import pandas as pd
import scipy.sparse as sparse
import numpy as np
import yaml
from typing import Tuple, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yml"

# Threshold for considering a rating as "relevant" (positive)
RELEVANCE_THRESHOLD = 3.5

class MovieDataset:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config["data"]["processed"]["base_dir"]
        self.ratings_path = self.config["data"]["processed"]["ratings"]
        self.movies_path = self.config["data"]["processed"]["movies"]
        
        # Get relevance threshold from config or use default
        self.relevance_threshold = self.config.get("data", {}).get("relevance_threshold", RELEVANCE_THRESHOLD)
        
        self.ratings = None
        self.movies = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

    def load_data(self):
        logger.info("Loading processed data...")
        self.ratings = pd.read_parquet(self.ratings_path)
        self.movies = pd.read_parquet(self.movies_path)
        
        # Create mappings
        unique_users = self.ratings['userId'].unique()
        unique_items = self.ratings['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {i: u for u, i in enumerate(unique_items)} # Wait, item_map should be item_id -> index
        self.item_map = {u: i for i, u in enumerate(unique_items)}
        
        self.reverse_user_map = {i: u for u, i in self.user_map.items()}
        self.reverse_item_map = {i: u for u, i in self.item_map.items()}
        
        logger.info(f"Loaded {len(unique_users)} users and {len(unique_items)} items.")

    def get_interaction_matrix(self) -> sparse.csr_matrix:
        if self.ratings is None:
            self.load_data()
            
        if hasattr(self, 'interaction_matrix') and self.interaction_matrix is not None:
            return self.interaction_matrix

        rows = [self.user_map[u] for u in self.ratings['userId']]
        cols = [self.item_map[i] for i in self.ratings['movieId']]
        data = self.ratings['rating'].values
        
        shape = (len(self.user_map), len(self.item_map))
        self.interaction_matrix = sparse.csr_matrix((data, (rows, cols)), shape=shape)
        return self.interaction_matrix

    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Split data into train and test sets using leave-k-out strategy per user.
        Only considers ratings >= relevance_threshold as positive interactions.
        Returns binary matrices (1 for positive, 0 for no interaction).
        """
        if self.ratings is None:
            self.load_data()
        
        # Filter only positive ratings (>= threshold)
        positive_ratings = self.ratings[self.ratings['rating'] >= self.relevance_threshold].copy()
        logger.info(f"Positive ratings (>= {self.relevance_threshold}): {len(positive_ratings)} / {len(self.ratings)} ({len(positive_ratings)/len(self.ratings)*100:.1f}%)")
        
        # Sort by timestamp for temporal split
        positive_ratings = positive_ratings.sort_values(['userId', 'timestamp'])
        
        shape = (len(self.user_map), len(self.item_map))
        
        # Leave-k-out split: for each user, leave out k most recent items for test
        k_test = max(1, int(test_ratio * 10))  # e.g., leave out 2 items if test_ratio=0.2
        
        train_rows, train_cols, train_data = [], [], []
        test_rows, test_cols, test_data = [], [], []
        
        for user_id, group in positive_ratings.groupby('userId'):
            user_idx = self.user_map[user_id]
            items = group['movieId'].values
            
            if len(items) <= k_test:
                # If user has too few items, put all in train
                for item_id in items:
                    item_idx = self.item_map[item_id]
                    train_rows.append(user_idx)
                    train_cols.append(item_idx)
                    train_data.append(1.0)  # Binary: 1 for positive
            else:
                # Last k items go to test, rest to train
                train_items = items[:-k_test]
                test_items = items[-k_test:]
                
                for item_id in train_items:
                    item_idx = self.item_map[item_id]
                    train_rows.append(user_idx)
                    train_cols.append(item_idx)
                    train_data.append(1.0)
                
                for item_id in test_items:
                    item_idx = self.item_map[item_id]
                    test_rows.append(user_idx)
                    test_cols.append(item_idx)
                    test_data.append(1.0)
        
        train_matrix = sparse.csr_matrix((train_data, (train_rows, train_cols)), shape=shape)
        test_matrix = sparse.csr_matrix((test_data, (test_rows, test_cols)), shape=shape)
        
        logger.info(f"Train interactions: {train_matrix.nnz}, Test interactions: {test_matrix.nnz}")
        
        return train_matrix, test_matrix

    def get_content_features(self) -> pd.DataFrame:
        if self.movies is None:
            self.load_data()
            
        # Prepare text content
        # Combine title, overview, genres, keywords, cast, crew
        # For simplicity, let's use title, overview, genres
        
        df = self.movies.copy()
        df['overview'] = df['overview'].fillna('')
        df['genres'] = df['genres'].fillna('')
        # If keywords/cast/crew are available and processed, add them
        
        # Simple text combination
        df['text_content'] = df['title'] + " " + df['genres'].str.replace('|', ' ') + " " + df['overview']
        
        # Filter only items that are in the interaction matrix
        valid_items = set(self.item_map.keys())
        df = df[df['movieId'].isin(valid_items)]
        
        # Reindex to match matrix indices
        df['item_idx'] = df['movieId'].map(self.item_map)
        df = df.sort_values('item_idx').set_index('item_idx')
        
        return df

if __name__ == "__main__":
    ds = MovieDataset()
    ds.load_data()
    mat = ds.get_interaction_matrix()
    logger.info(f"Interaction matrix shape: {mat.shape}")
    content = ds.get_content_features()
    logger.info(f"Content features shape: {content.shape}")
