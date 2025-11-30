import os
import pandas as pd
import yaml
import ast
import numpy as np
from src.data.kaggle_loader import KaggleLoader

CONFIG_PATH = "config/config.yml"

class Preprocessor:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.loader = KaggleLoader(config_path)
        self.processed_dir = self.config["data"]["processed"]["base_dir"]
        os.makedirs(self.processed_dir, exist_ok=True)

    def process_movielens(self):
        print("Processing MovieLens data...")
        ratings = self.loader.load_movielens_ratings()
        movies = self.loader.load_movielens_movies()
        
        # Filter ratings
        min_user_ratings = self.config["data"]["split"]["min_ratings_per_user"]
        min_item_ratings = self.config["data"]["split"]["min_ratings_per_item"]
        
        user_counts = ratings['userId'].value_counts()
        item_counts = ratings['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        
        ratings = ratings[ratings['userId'].isin(valid_users) & ratings['movieId'].isin(valid_items)]
        
        print(f"Filtered ratings shape: {ratings.shape}")
        
        # Save ratings
        ratings_path = self.config["data"]["processed"]["ratings"]
        ratings.to_parquet(ratings_path, index=False)
        print(f"Saved ratings to {ratings_path}")
        
        return ratings, movies

    def process_metadata(self, ml_movies):
        print("Processing Metadata...")
        metadata_dict = self.loader.load_movies_metadata()
        
        metadata = metadata_dict["metadata"]
        links = metadata_dict["links"]
        credits = metadata_dict["credits"]
        keywords = metadata_dict["keywords"]
        
        # Clean links
        links = links.dropna(subset=['tmdbId'])
        links['tmdbId'] = links['tmdbId'].astype(int)
        
        # Clean metadata
        # Convert 'id' to int, handle errors
        metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
        metadata = metadata.dropna(subset=['id'])
        metadata['id'] = metadata['id'].astype(int)
        
        # Merge links with metadata on tmdbId (metadata 'id' is tmdbId)
        # links: movieId, imdbId, tmdbId
        # metadata: id (tmdbId), ...
        
        merged_meta = pd.merge(links, metadata, left_on='tmdbId', right_on='id', how='inner')
        
        # Merge with credits and keywords if needed (on id)
        # Note: credits and keywords also use tmdbId as 'id'
        
        credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
        credits = credits.dropna(subset=['id']).astype({'id': int})
        
        keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
        keywords = keywords.dropna(subset=['id']).astype({'id': int})
        
        merged_meta = pd.merge(merged_meta, credits, on='id', how='left')
        merged_meta = pd.merge(merged_meta, keywords, on='id', how='left')
        
        # Merge with MovieLens movies to keep only relevant ones
        # ml_movies: movieId, title, genres
        final_movies = pd.merge(ml_movies, merged_meta, on='movieId', how='left')
        
        # Resolve suffixes
        if 'title_x' in final_movies.columns:
            final_movies['title'] = final_movies['title_x']
            final_movies = final_movies.drop(columns=['title_x', 'title_y'], errors='ignore')
            
        if 'genres_x' in final_movies.columns:
            final_movies['genres'] = final_movies['genres_x']
            final_movies = final_movies.drop(columns=['genres_x', 'genres_y'], errors='ignore')
        
        # Save movies
        movies_path = self.config["data"]["processed"]["movies"]
        final_movies.to_parquet(movies_path, index=False)
        print(f"Saved movies to {movies_path}")
        
        return final_movies

    def run(self):
        ratings, ml_movies = self.process_movielens()
        self.process_metadata(ml_movies)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
