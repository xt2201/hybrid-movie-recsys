import pandas as pd
import scipy.sparse as sparse
import numpy as np
import yaml
from typing import Tuple, Dict

CONFIG_PATH = "config/config.yml"

class MovieDataset:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config["data"]["processed"]["base_dir"]
        self.ratings_path = self.config["data"]["processed"]["ratings"]
        self.movies_path = self.config["data"]["processed"]["movies"]
        
        self.ratings = None
        self.movies = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

    def load_data(self):
        print("Loading processed data...")
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
        
        print(f"Loaded {len(unique_users)} users and {len(unique_items)} items.")

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
        if self.ratings is None:
            self.load_data()
            
        # Simple random split
        # For a real scenario, we should split by time or leave-one-out per user
        # Here we do a global random split for simplicity as a baseline
        
        n_ratings = len(self.ratings)
        test_size = int(n_ratings * test_ratio)
        train_size = n_ratings - test_size
        
        # Shuffle indices
        indices = np.random.permutation(n_ratings)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Create train matrix
        train_ratings = self.ratings.iloc[train_indices]
        rows_train = [self.user_map[u] for u in train_ratings['userId']]
        cols_train = [self.item_map[i] for i in train_ratings['movieId']]
        data_train = train_ratings['rating'].values
        shape = (len(self.user_map), len(self.item_map))
        train_matrix = sparse.csr_matrix((data_train, (rows_train, cols_train)), shape=shape)
        
        # Create test matrix
        test_ratings = self.ratings.iloc[test_indices]
        rows_test = [self.user_map[u] for u in test_ratings['userId']]
        cols_test = [self.item_map[i] for i in test_ratings['movieId']]
        data_test = test_ratings['rating'].values
        test_matrix = sparse.csr_matrix((data_test, (rows_test, cols_test)), shape=shape)
        
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
    print(f"Interaction matrix shape: {mat.shape}")
    content = ds.get_content_features()
    print(f"Content features shape: {content.shape}")
