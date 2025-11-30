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
    """Calculate Precision@K: fraction of recommended items that are relevant."""
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)
    if len(recommended_items) == 0:
        return 0.0
    num_hits = sum(1 for item in recommended_items if item in relevant_items)
    return num_hits / len(recommended_items)

def recall_at_k(recommended_items, relevant_items, k):
    """Calculate Recall@K: fraction of relevant items that were recommended."""
    recommended_items = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    if len(relevant_items) == 0:
        return 0.0
    num_hits = len(recommended_items & relevant_items)
    return num_hits / len(relevant_items)

def hit_rate_at_k(recommended_items, relevant_items, k):
    """Calculate Hit Rate@K: 1 if any relevant item in top-K, else 0."""
    recommended_items = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    return 1.0 if len(recommended_items & relevant_items) > 0 else 0.0

def ndcg_at_k(recommended_items, relevant_items, k):
    """Calculate NDCG@K."""
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)
    
    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position starts at 1
    
    # Ideal DCG
    n_relevant = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, train_matrix, test_matrix, k=10, n_users=500):
    """
    Evaluate model using test matrix where entries represent relevant items.
    The test matrix should only contain positive interactions (ratings >= threshold).
    """
    print(f"Evaluating model on up to {n_users} users...")
    
    # Get users who have relevant items in test set
    test_users = np.unique(test_matrix.nonzero()[0])
    
    # Sample users
    if len(test_users) > n_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, n_users, replace=False)
    
    print(f"  Found {len(test_users)} test users with relevant items")
    
    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []
    
    for user_id in tqdm(test_users, desc="Evaluating"):
        # Get ground truth: items in test set are the relevant items
        relevant_items = test_matrix[user_id].indices
        
        if len(relevant_items) == 0:
            continue
        
        # Get recommendations
        try:
            if isinstance(model, SVDRecommender):
                ids, scores = model.recommend(user_id, train_matrix, N=k)
            else:
                # Hybrid or others
                recs = model.recommend(user_id, N=k, user_item_matrix=train_matrix)
                ids = [x[0] for x in recs]
        except Exception as e:
            continue
        
        precisions.append(precision_at_k(ids, relevant_items, k))
        recalls.append(recall_at_k(ids, relevant_items, k))
        hit_rates.append(hit_rate_at_k(ids, relevant_items, k))
        ndcgs.append(ndcg_at_k(ids, relevant_items, k))
    
    metrics = {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls),
        f'HitRate@{k}': np.mean(hit_rates),
        f'NDCG@{k}': np.mean(ndcgs)
    }
    
    print(f"\nResults ({len(precisions)} valid users):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics

def main():
    np.random.seed(42)
    
    ds = MovieDataset()
    ds.load_data()
    train_matrix, test_matrix = ds.get_train_test_split(test_ratio=0.2)
    
    print("\n" + "="*50)
    print("Evaluating SVD/ALS Model...")
    print("="*50)
    svd_model = SVDRecommender()
    svd_model.fit(train_matrix)
    evaluate_model(svd_model, train_matrix, test_matrix, k=10, n_users=500)
    
    print("\n" + "="*50)
    print("Evaluating Hybrid Model...")
    print("="*50)
    
    hybrid_model = HybridRecommender()
    hybrid_model.dataset = ds  # Reuse loaded dataset
    hybrid_model.cf_model.fit(train_matrix)
    
    content_df = ds.get_content_features()
    hybrid_model.content_model.fit(content_df)
    hybrid_model.is_fitted = True
    
    evaluate_model(hybrid_model, train_matrix, test_matrix, k=10, n_users=500)

if __name__ == "__main__":
    main()
