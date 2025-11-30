# Hybrid Movie Recommendation System

A comprehensive movie recommendation system combining Collaborative Filtering, Content-based Filtering, and Large Language Models (LLM).

## Features

*   **Hybrid Recommendation Engine**: Combines SVD (Collaborative Filtering) and TF-IDF (Content-based) for robust recommendations.
*   **LLM Integration**: Uses **Qwen2.5-1.5B-Instruct** for:
    *   Natural Language Query Parsing (extracting genres, mood, etc.).
    *   Re-ranking candidates based on query relevance.
    *   Generating natural language explanations for recommendations.
*   **API Serving**: Fast and scalable API built with **FastAPI**.
*   **MLOps**: Integrated with **Weights & Biases** for experiment tracking.
*   **Containerization**: Docker support for easy deployment.

## Project Structure

```bash
hybrid-movie-recsys/
├── config/                 # Configuration files
├── data/                   # Data directory (raw & processed)
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for EDA
├── scripts/                # Utility scripts (download, train, eval)
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data/               # Data loading and preprocessing
│   ├── llm/                # LLM integration (Qwen)
│   ├── models/             # CF and Content models
│   └── recommender/        # Hybrid recommender logic
├── tests/                  # Unit tests
├── Dockerfile              # Docker configuration
├── pyproject.toml          # Poetry dependencies
├── requirements.txt        # Pip requirements
└── README.md               # This file
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/xt2201/hybrid-movie-recsys.git
    cd hybrid-movie-recsys
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment**:
    Create a `.env` file (see `.env.example` if available) with necessary keys (WANDB, HF_TOKEN, etc.).

## Data Setup

The system uses MovieLens and TMDB datasets from Kaggle.

1.  **Download Datasets**:
    ```bash
    python scripts/download_kaggle_datasets.py
    ```
    *Note: Ensure you have `kaggle.json` in `~/.kaggle/` or set `KAGGLE_CONFIG_DIR`.*

2.  **Preprocess Data**:
    ```bash
    PYTHONPATH=. python src/data/preprocess.py
    ```

## Usage

### Running the API

Start the FastAPI server:

```bash
python -m uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`. You can access the Swagger UI at `http://localhost:8000/docs`.

### Testing the API

Run the test script to verify endpoints:

```bash
python scripts/test_api.py
```

### Training & Evaluation

*   **Evaluate Baselines**:
    ```bash
    PYTHONPATH=. python scripts/evaluate.py
    ```
*   **Train Hybrid Model**:
    ```bash
    PYTHONPATH=. python scripts/train_hybrid.py
    ```
    *This script also uploads model checkpoints to Hugging Face Hub (`xt2201/hybrid-movie-recsys`) if `HF_TOKEN` is set.*

*   **Evaluate LLM Components**:
    ```bash
    PYTHONPATH=. python scripts/evaluate_llm.py
    ```

### Full Pipeline Notebook

Run `notebooks/02_full_pipeline.ipynb` for an end-to-end demonstration, including data download, training, evaluation, inference, and model comparison.

## Docker

Build and run with Docker:

```bash
docker build -t hybrid-recsys .
docker run -p 8000:8000 hybrid-recsys
```

## License

[MIT](LICENSE)
