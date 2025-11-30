import yaml
import wandb
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender.hybrid import HybridRecommender
from src.data.dataset import MovieDataset
from scripts.evaluate import evaluate_model

CONFIG_PATH = "config/config.yml"

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize W&B
    if config["logging"]["wandb"]["enabled"]:
        wandb.init(
            project=config["logging"]["wandb"]["project"],
            entity=config["logging"]["wandb"]["entity"],
            tags=config["logging"]["wandb"]["tags"],
            config=config
        )
        
    print("Initializing Hybrid Recommender...")
    recsys = HybridRecommender(CONFIG_PATH)
    
    # Load data
    ds = MovieDataset(CONFIG_PATH)
    train_matrix, test_matrix = ds.get_train_test_split(test_ratio=config["data"]["split"]["test_ratio"])
    
    # We need to manually fit components to use the split
    print("Training CF Model (SVD)...")
    recsys.cf_model.fit(train_matrix)
    
    print("Training Content Model...")
    content_df = ds.get_content_features()
    recsys.content_model.fit(content_df)
    
    recsys.is_fitted = True
    
    # Evaluate
    print("Evaluating...")
    # We use a custom evaluation function that logs to W&B
    k = 10
    n_users = config["evaluation"]["eval_users_sample"]
    
    # Reuse evaluate_model logic but capture metrics
    # For simplicity, let's just call evaluate_model and parse output or modify it?
    # Better to copy-paste logic or import and modify.
    # Let's write a simple eval loop here for W&B.
    
    test_users = np.unique(test_matrix.nonzero()[0])
    if len(test_users) > n_users:
        test_users = np.random.choice(test_users, n_users, replace=False)
        
    precisions = []
    recalls = []
    
    for user_id in test_users:
        relevant_items = test_matrix[user_id].indices
        if len(relevant_items) == 0:
            continue
            
        try:
            recs = recsys.recommend(user_id, N=k, user_item_matrix=train_matrix)
            ids = [x[0] for x in recs]
            
            # Precision
            hits = sum(1 for i in ids if i in relevant_items)
            precisions.append(hits / k)
            
            # Recall
            recalls.append(hits / len(relevant_items))
        except Exception:
            continue
            
    p_k = np.mean(precisions)
    r_k = np.mean(recalls)
    
    print(f"Precision@{k}: {p_k:.4f}")
    print(f"Recall@{k}: {r_k:.4f}")
    
    if config["logging"]["wandb"]["enabled"]:
        wandb.log({
            f"precision@{k}": p_k,
            f"recall@{k}": r_k
        })
        wandb.finish()

    # Save and Upload to HF
    print("Saving models...")
    output_dir = config["project"]["output_dir"]
    recsys.cf_model.save(f"{output_dir}/cf_model.pkl")
    recsys.content_model.save(f"{output_dir}/content_model.pkl")
    
    # Upload to HF
    try:
        from huggingface_hub import HfApi
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        repo_id = "xt2201/hybrid-movie-recsys"
        
        if hf_token:
            print(f"Uploading to Hugging Face Hub: {repo_id}...")
            api = HfApi(token=hf_token)
            
            # Create repo if not exists (private by default if not specified, but let's assume it exists or public)
            api.create_repo(repo_id=repo_id, exist_ok=True)
            
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                path_in_repo="checkpoints",
                ignore_patterns=["*.parquet", "*.csv"] # Don't upload data if accidentally in output
            )
            print("Upload complete.")
        else:
            print("HF_TOKEN not found. Skipping upload.")
            
    except Exception as e:
        print(f"Error uploading to HF: {e}")

if __name__ == "__main__":
    main()
