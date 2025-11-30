import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from src.data.dataset import MovieDataset
from src.models.cf.svd import SVDRecommender
from src.models.content.tfidf import ContentRecommender

# Mock config or use default
CONFIG_PATH = "config/config.yml"

def test_dataset_loading():
    ds = MovieDataset(CONFIG_PATH)
    # We assume data is already processed in Sprint 1
    # If not, this test might fail or we should mock loading
    # For integration test, we use real data
    ds.load_data()
    assert ds.ratings is not None
    assert ds.movies is not None
    assert len(ds.user_map) > 0
    assert len(ds.item_map) > 0

def test_svd_recommender():
    # Create dummy data
    users = [0, 0, 1, 1, 2]
    items = [0, 1, 0, 2, 1]
    ratings = [5, 4, 4, 5, 3]
    mat = sparse.csr_matrix((ratings, (users, items)), shape=(3, 3))
    
    model = SVDRecommender(CONFIG_PATH)
    model.factors = 2 # Override for small data
    model.model.n_components = 2
    
    model.fit(mat)
    assert model.user_factors is not None
    assert model.item_factors is not None
    
    ids, scores = model.recommend(0, mat, N=2)
    assert len(ids) <= 2

def test_content_recommender():
    # Create dummy content
    data = {
        'text_content': [
            "Action movie with explosions",
            "Romantic comedy with love",
            "Action sci-fi space"
        ]
    }
    df = pd.DataFrame(data)
    
    model = ContentRecommender(CONFIG_PATH)
    model.fit(df)
    assert model.tfidf_matrix is not None
    
    ids, scores = model.recommend(0, N=2)
    assert len(ids) <= 2
