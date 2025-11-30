import os
import yaml
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

CONFIG_PATH = "config/config.yml"

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    kaggle_cfg = cfg["data"]["kaggle"]
    raw_dir = kaggle_cfg["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    # Download MovieLens
    ml_cfg = kaggle_cfg["movielens"]
    dataset = ml_cfg["dataset"]
    subdir = ml_cfg["subdir"]
    out_dir = os.path.join(raw_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Downloading {dataset} to {out_dir}...")
    try:
        api.dataset_download_files(dataset, path=out_dir, unzip=True)
        print(f"Successfully downloaded {dataset}")
    except Exception as e:
        print(f"Error downloading {dataset}: {e}")

    # Download Movies Metadata
    meta_cfg = kaggle_cfg["movies_metadata"]
    dataset = meta_cfg["dataset"]
    subdir = meta_cfg["subdir"]
    out_dir = os.path.join(raw_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {dataset} to {out_dir}...")
    try:
        api.dataset_download_files(dataset, path=out_dir, unzip=True)
        print(f"Successfully downloaded {dataset}")
    except Exception as e:
        print(f"Error downloading {dataset}: {e}")

if __name__ == "__main__":
    main()
