import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import pickle
import os

CONFIG_PATH = "config/config.yml"

class ContentRecommender:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.content_config = self.config["model"]["content"]
        self.max_features = self.content_config["max_features"]
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        self.tfidf_matrix = None
        self.item_ids = None # List of item IDs corresponding to rows

    def fit(self, content_df: pd.DataFrame):
        print("Training Content-based model (TF-IDF)...")
        # content_df should have 'text_content' column and index as item_idx (or we keep track)
        self.item_ids = content_df.index.tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(content_df['text_content'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def recommend(self, item_idx: int, N: int = 10):
        # Recommend items similar to item_idx
        # item_idx is the index in the interaction matrix, which matches our self.item_ids order if sorted
        
        # Find the row index in tfidf_matrix corresponding to item_idx
        try:
            row_idx = self.item_ids.index(item_idx)
        except ValueError:
            print(f"Item {item_idx} not found in content model.")
            return [], []
            
        # Compute cosine similarity with all other items
        # To optimize, we can compute only for this item
        cosine_sim = cosine_similarity(self.tfidf_matrix[row_idx], self.tfidf_matrix).flatten()
        
        # Get top N
        # argsort returns indices in ascending order, so we take last N+1 (including self)
        top_indices = cosine_sim.argsort()[-(N+1):][::-1]
        
        # Exclude self
        top_indices = [i for i in top_indices if i != row_idx][:N]
        
        recommended_item_idxs = [self.item_ids[i] for i in top_indices]
        scores = [cosine_sim[i] for i in top_indices]
        
        return recommended_item_idxs, scores

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'item_ids': self.item_ids
            }, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.item_ids = data['item_ids']
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    from src.data.dataset import MovieDataset
    ds = MovieDataset()
    content_df = ds.get_content_features()
    
    model = ContentRecommender()
    model.fit(content_df)
    
    # Test recommendation
    item_idx = 0
    ids, scores = model.recommend(item_idx)
    print(f"Recommendations similar to item {item_idx}: {ids}")
