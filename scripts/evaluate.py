import numpy as np
import yaml
from tqdm import tqdm
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MovieDataset
from src.models.cf.svd import SVDRecommender
from src.recommender.hybrid import HybridRecommender

CONFIG_PATH = "config/config.yml"

def precision_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)
    num_hits = sum(1 for item in recommended_items if item in relevant_items)
    return num_hits / k

def recall_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)
    num_hits = sum(1 for item in recommended_items if item in relevant_items)
    return num_hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

def evaluate_model(model, train_matrix, test_matrix, k=10, n_users=100):
    print(f"Evaluating model on {n_users} users...")
    
    # Get users who have interactions in test set
    test_users = np.unique(test_matrix.nonzero()[0])
    
    # Sample users
    if len(test_users) > n_users:
        test_users = np.random.choice(test_users, n_users, replace=False)
    
    precisions = []
    recalls = []
    
    for user_id in tqdm(test_users):
        # Get ground truth
        relevant_items = test_matrix[user_id].indices
        
        if len(relevant_items) == 0:
            continue
            
        # Get recommendations
        # Note: For SVD, we pass the train_matrix as user_items to mask already seen items
        # For Hybrid, it handles it internally or we need to pass it?
        # Hybrid recommend signature: recommend(user_id, N)
        # SVD recommend signature: recommend(user_id, user_item_matrix, N)
        
        try:
            if isinstance(model, SVDRecommender):
                ids, scores = model.recommend(user_id, train_matrix, N=k)
            else:
                # Hybrid or others
                recs = model.recommend(user_id, N=k)
                ids = [x[0] for x in recs]
        except Exception as e:
            print(f"Error recommending for user {user_id}: {e}")
            continue
            
        precisions.append(precision_at_k(ids, relevant_items, k))
        recalls.append(recall_at_k(ids, relevant_items, k))
        
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}: {np.mean(recalls):.4f}")

def main():
    ds = MovieDataset()
    train_matrix, test_matrix = ds.get_train_test_split(test_ratio=0.2)
    
    print("Evaluating SVD Model...")
    svd_model = SVDRecommender()
    svd_model.fit(train_matrix)
    evaluate_model(svd_model, train_matrix, test_matrix, k=10, n_users=100)
    
    print("\nEvaluating Hybrid Model...")
    # Hybrid needs to be fitted with full data or train data?
    # Hybrid uses dataset internally. We should probably modify Hybrid to accept train matrix.
    # But for now, let's just instantiate it. It will load data again from disk.
    # This is not ideal for split evaluation because Hybrid will see full data if it reloads.
    # We need to hack Hybrid to use our train_matrix.
    
    hybrid_model = HybridRecommender()
    # Inject train matrix
    hybrid_model.dataset.ratings = None # Force reload? No, we want to inject.
    # Hybrid.fit() calls dataset.load_data() and get_interaction_matrix().
    # We can manually set the interaction matrix in hybrid's dataset.
    
    # But hybrid also uses content model which uses all movies. That's fine (content features are static).
    # The issue is CF part of Hybrid.
    
    # Let's manually fit components of Hybrid
    hybrid_model.dataset.load_data() # Load maps
    hybrid_model.cf_model.fit(train_matrix) # Fit CF on train
    
    content_df = hybrid_model.dataset.get_content_features()
    hybrid_model.content_model.fit(content_df)
    hybrid_model.is_fitted = True
    
    # Evaluate
    evaluate_model(hybrid_model, train_matrix, test_matrix, k=10, n_users=100)

if __name__ == "__main__":
    main()
