import os
import pandas as pd
import yaml
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

class KaggleLoader:
    def __init__(self, config_path: str = "config/config.yml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.kaggle_cfg = self.config["data"]["kaggle"]
        self.raw_dir = self.kaggle_cfg["raw_dir"]

    def load_movielens_ratings(self) -> pd.DataFrame:
        ml_cfg = self.kaggle_cfg["movielens"]
        path = os.path.join(self.raw_dir, ml_cfg["subdir"], ml_cfg["ratings_file"])
        logger.info(f"Loading ratings from {path}...")
        return pd.read_csv(path)

    def load_movielens_movies(self) -> pd.DataFrame:
        ml_cfg = self.kaggle_cfg["movielens"]
        path = os.path.join(self.raw_dir, ml_cfg["subdir"], ml_cfg["movies_file"])
        logger.info(f"Loading movies from {path}...")
        return pd.read_csv(path)

    def load_movies_metadata(self) -> Dict[str, pd.DataFrame]:
        meta_cfg = self.kaggle_cfg["movies_metadata"]
        subdir = meta_cfg["subdir"]
        files = meta_cfg["files"]
        
        data = {}
        for key, filename in files.items():
            path = os.path.join(self.raw_dir, subdir, filename)
            logger.info(f"Loading {key} from {path}...")
            # Some files might have bad lines or mixed types, handle with care
            try:
                if key == "metadata":
                    # metadata often has mixed types and parsing errors
                    data[key] = pd.read_csv(path, low_memory=False)
                else:
                    data[key] = pd.read_csv(path)
            except Exception as e:
                logger.info(f"Error loading {path}: {e}")
        
        return data

if __name__ == "__main__":
    loader = KaggleLoader()
    ratings = loader.load_movielens_ratings()
    logger.info(f"Ratings shape: {ratings.shape}")
    
    movies = loader.load_movielens_movies()
    logger.info(f"Movies shape: {movies.shape}")
    
    metadata = loader.load_movies_metadata()
    for k, v in metadata.items():
        logger.info(f"{k} shape: {v.shape}")
