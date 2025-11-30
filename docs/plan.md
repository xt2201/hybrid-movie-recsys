**Hybrid Movie Recommendation System** (MovieLens + TMDB/IMDb)
**ML (CF + Content-based)** + **LLM Qwen3-1.7B**
---

## 1. Mục tiêu & kiến trúc hệ thống

**Mục tiêu:**

* Hệ thống gợi ý phim:

  * Dựa trên lịch sử rating (MovieLens) → **Collaborative Filtering (CF)**.
  * Dựa trên metadata phim (TMDB/IMDb từ Kaggle) → **Content-based**.
  * Dùng **Qwen3-1.7B** để:

    * Hiểu truy vấn tự nhiên (VN/EN).
    * Re-rank danh sách candidate.
    * Sinh explanation cho user.

**Kiến trúc high-level:**

**Offline:**

1. ETL từ Kaggle:

   * MovieLens ratings + movies.
   * Metadata TMDB/IMDb (overview, genres, cast, crew…).
2. Feature engineering:

   * User–item interactions.
   * Content features (TF-IDF / embedding).
3. Train:

   * CF model (ALS/BPR/Neural CF).
   * Content-based model.
4. Tích hợp LLM:

   * Module parse query.
   * Module re-rank.
   * Module explanation.
5. Log experiment lên **W&B**.

**Online:**

1. User gọi API (by user_id hoặc by text query).
2. CF → top-N candidate.
3. Content-based → bổ sung/bù CF.
4. LLM Qwen:

   * Parse truy vấn → preference.
   * Re-rank candidate.
   * Sinh explanation.
5. API trả JSON: list phim + score + lý do.

---

## 2. Dataset & Kaggle (dùng `kaggle.json`)

Bạn đã có `kaggle.json`, nên:

* Đặt file tại: `~/.kaggle/kaggle.json` (Linux/Mac)
  hoặc `C:\Users\<User>\.kaggle\kaggle.json` (Windows).
* Phân quyền: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac) cho chắc.

**Đề xuất bộ dataset Kaggle:**

1. **MovieLens**

   * `movielens-20m` hoặc `movielens-25m` (ratings + movies).
2. **Metadata phim (TMDB/IMDb)**

   * `The Movies Dataset` (đã có nhiều file: metadata, credits, keywords, links).
   * Hoặc `TMDB 5000 Movie Dataset` để thử nghiệm nhanh hơn.

**Data flow (sử dụng Kaggle CLI):**

* Script `scripts/download_kaggle_datasets.py` sẽ dùng:

  ```bash
  kaggle datasets download -d <owner>/<dataset> -p data/raw/<subdir> --unzip
  ```
* Không cần env cho Kaggle nữa, Kaggle tự dùng `kaggle.json`.

---

## 3. Agile – Vision, Epics, User Stories

### Product Vision

> “Xây dựng hệ thống gợi ý phim cá nhân hóa, hiểu ngôn ngữ tự nhiên, kết hợp ML + LLM, dễ mở rộng và theo dõi bằng MLOps.”

### Epics

1. **EPIC-DATA-KAGGLE** – Lấy và chuẩn hóa data từ Kaggle.
2. **EPIC-ML-CF** – CF model từ MovieLens.
3. **EPIC-ML-CONTENT** – Content-based từ TMDB/IMDb.
4. **EPIC-LLM-QWEN** – Tích hợp Qwen3-1.7B (query + re-rank + explain).
5. **EPIC-API-SERVING** – FastAPI/gRPC phục vụ recommendation.
6. **EPIC-MLOPS** – Logging, config, training pipeline, W&B, tests.

### User Stories (ví dụ)

* **US-01**: Là user, tôi muốn xem **top-10 phim gợi ý** dựa trên lịch sử rating.
* **US-02**: Là user, tôi muốn gõ truy vấn dạng “phim sci-fi buồn giống Interstellar” và nhận được gợi ý phù hợp.
* **US-03**: Là DS, tôi muốn config experiment qua **config.yml**, không sửa code.
* **US-04**: Là DS, tôi muốn tất cả training run được log lên **W&B**.

---

## 4. Kế hoạch Sprint (đã chỉnh theo Kaggle + kaggle.json)

### Sprint 0 – Project setup & tooling

**Mục tiêu:** Repo skeleton, virtual env, tích hợp Kaggle (dùng `kaggle.json`), W&B, HF.

**Tasks:**

* Khởi tạo repo & env:

  * `pyproject.toml` (poetry) hoặc `requirements.txt`.
  * Thư mục `src`, `scripts`, `config`, `data`, `notebooks`, `tests`.
* Thiết lập W&B & HuggingFace:

  * Thêm **WANDB_API_KEY**, **HF_TOKEN** vào `.env`.
* Kaggle:

  * Move `kaggle.json` → `~/.kaggle/kaggle.json`.
  * Viết `scripts/download_kaggle_datasets.py`.
* Tạo `config/config.yml` skeleton.

---

### Sprint 1 – Data từ Kaggle → schema chuẩn

**Mục tiêu:** Lấy và chuẩn hóa dữ liệu từ Kaggle vào `data/processed`.

**Tasks:**

1. `scripts/download_kaggle_datasets.py`:

   * Đọc `config.yml` để biết:

     * Dataset MovieLens (e.g. `grouplens/movielens-20m-dataset`).
     * Dataset TMDB/metadata (e.g. `rounakbanik/the-movies-dataset`).
   * Gọi `kaggle datasets download` cho từng dataset.
2. `src/data/kaggle_loader.py`:

   * Hàm tải CSV từ `data/raw/...`.
3. `src/data/preprocess.py`:

   * Chuẩn hóa MovieLens:

     * Filter user/item có min số rating.
   * Join với metadata:

     * Qua `links.csv` (movieId ↔ tmdbId/imdbId) nếu dùng The Movies Dataset.
   * Sinh các bảng:

     * `ratings.parquet`, `movies.parquet`, `movies_metadata.parquet`, `movies_cast_crew.parquet`.
4. EDA:

   * `notebooks/01_eda_kaggle_datasets.ipynb`.

---

### Sprint 2 – Feature engineering & baseline models

**Mục tiêu:** Có CF + content-based chạy offline, metric cơ bản.

**Tasks:**

* **Feature engineering:**

  * Tạo `user_item_matrix` cho CF.
  * Tạo text field cho content:

    * `genres + overview + tagline + keywords + cast + crew`.
* **CF model:**

  * Dùng `implicit` (ALS/BPR) hoặc PyTorch:

    * Train trên interactions binary (rating ≥ threshold).
  * Log hyperparams + metric vào W&B.
* **Content-based model:**

  * TF-IDF trên text field (hoặc Sentence-BERT / embedding khác).
  * Xây index (cosine similarity).
* **Hybrid baseline (chưa LLM):**

  * `src/recommender/hybrid.py`:

    ```python
    score = alpha_cf * score_cf + (1 - alpha_cf) * score_content
    ```
* Evaluation:

  * NDCG@K, Recall@K trên tập test.

---

### Sprint 3 – Tích hợp LLM Qwen3-1.7B

**Mục tiêu:** Dùng Qwen cho query → preference + re-ranking + explanation.

**Tasks:**

1. Module LLM:

   * `src/llm/qwen_client.py`:

     * Load Qwen từ HuggingFace (ví dụ: `Qwen/Qwen2.5-1.7B-Instruct`).
   * Sử dụng config `llm` trong `config.yml`.
2. `src/llm/query_parser.py`:

   * Input: text query user.
   * Output: structured preference:

     ```json
     {
       "must_genres": ["Sci-Fi"],
       "avoid_genres": ["Horror"],
       "mood": "sad",
       "similar_titles": ["Interstellar"],
       "year_from": 2000,
       "year_to": 2025
     }
     ```
   * Prompt template lưu ở `config/prompts/query_parser.txt`.
3. `src/llm/reranker.py`:

   * Input:

     * Candidate list (top-N CF + content).
     * User context (history).
     * Parsed query.
   * Ask Qwen cho **ranking hoặc scoring** mới.
   * Prompt ở `config/prompts/reranker.txt`.
4. `src/llm/explainer.py`:

   * Sinh explanation ngắn (1–2 câu) cho mỗi phim.
   * Prompt ở `config/prompts/explainer.txt`.
5. Evaluate:

   * Offline: NDCG@K sau khi re-rank.
   * Qualitative: check vài example explanation.

---

### Sprint 4 – API & Serving

**Mục tiêu:** Cho phép gọi gợi ý qua HTTP.

**Tasks:**

* `src/api/main.py` với **FastAPI**:

  * `POST /recommend/user/{user_id}`:

    * Input: user_id, optional: num_items.
    * Flow: CF → Content → (optional LLM re-rank) → trả list phim.
  * `POST /recommend/query`:

    * Input: text query, optional: user_id.
    * Flow: query_parser → candidate (CF+content) → LLM re-rank → return.
* `src/api/schemas.py`: Pydantic models cho request/response.
* `scripts/run_api.py`: chạy FastAPI (uvicorn).
* Dockerfile basic.

---

### Sprint 5 – MLOps & tối ưu

**Mục tiêu:** Chuẩn hóa training pipeline, logging, testing.

**Tasks:**

* Training pipeline:

  * `scripts/train_cf.py`, `train_content.py`, `train_hybrid.py`.
  * Đọc config từ `config.yml`.
* W&B:

  * Tích hợp logging trong training scripts.
  * (Optional) W&B sweeps cho hyperparameter search.
* Logging & monitoring:

  * `src/utils/logging.py` (log dạng JSON).
* Tests:

  * `tests/` cho data, models, llm integration, api.
* (Optional) Model registry:

  * Lưu model artifact (pickle, torch, onnx) trong `outputs/checkpoints`.

---

## 5. Cấu trúc thư mục

```bash
./
├── README.md
├── pyproject.toml              # hoặc requirements.txt
├── .gitignore
├── .env                        # chứa wandb, HF, DB... (KHÔNG chứa Kaggle)
├── .env.example
├── config/
│   ├── config.yml              # config tổng
│   ├── data.yml                # optional: tách data config
│   ├── model_cf.yml            # optional
│   ├── model_content.yml
│   ├── model_llm.yml
│   └── prompts/
│       ├── query_parser.md
│       ├── reranker.md
│       └── explainer.md
├── data/
│   ├── raw/
│   │   ├── movielens/          # download từ Kaggle
│   │   └── movies_metadata/    # The Movies Dataset / TMDB 5000...
│   ├── interim/
│   └── processed/
│       ├── ratings.parquet
│       ├── movies.parquet
│       ├── movies_metadata.parquet
│       └── movie_features.parquet
├── notebooks/
│   ├── 01_eda_kaggle_datasets.ipynb
│   ├── 02_cf_baseline.ipynb
│   └── 03_llm_qwen_prototype.ipynb
├── scripts/
│   ├── download_kaggle_datasets.py
│   ├── build_features_from_kaggle.py
│   ├── train_cf.py
│   ├── train_content.py
│   ├── train_hybrid.py
│   ├── evaluate.py
│   └── run_api.py
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py            # đọc config.yml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── kaggle_loader.py
│   │   ├── movielens.py
│   │   ├── movies_metadata.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cf/
│   │   │   ├── als.py
│   │   │   ├── bpr.py
│   │   │   └── neural_cf.py
│   │   ├── content/
│   │   │   ├── tfidf.py
│   │   │   └── embedding_model.py
│   │   └── utils.py
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── hybrid.py
│   │   └── ranking.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── qwen_client.py
│   │   ├── query_parser.py
│   │   ├── reranker.py
│   │   └── explainer.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── wandb_utils.py
│       └── metrics.py
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_models_cf.py
    ├── test_models_content.py
    ├── test_llm_integration.py
    └── test_api.py
```

---

## 6. `.env` (không dùng cho Kaggle nữa)

Chỉ cần cho W&B, HF, DB, LLM, API:

```bash
# === General ===
ENV=dev
PYTHONPATH=./src

# === Weights & Biases ===
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=hybrid-movie-recsys
WANDB_ENTITY=your_wandb_entity
WANDB_MODE=online

# === HuggingFace ===
HF_TOKEN=your_hf_token_here
HF_HUB_CACHE=./.cache/huggingface

# === Database (optional) ===
DB_HOST=localhost
DB_PORT=5432
DB_NAME=movies
DB_USER=postgres
DB_PASSWORD=postgres

# === LLM & Serving ===
LLM_MODEL_NAME=Qwen/Qwen2.5-1.7B-Instruct
LLM_DEVICE=cuda:0
LLM_MAX_NEW_TOKENS=128

# === API ===
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 7. `config/config.yml` (updated, có Kaggle nhưng không chứa token)

```yaml
project:
  name: "hybrid-movie-recsys"
  seed: 42
  output_dir: "outputs/"

logging:
  level: "INFO"
  log_dir: "logs/"
  wandb:
    enabled: true
    project: "hybrid-movie-recsys"
    entity: "your_wandb_entity"
    tags: ["kaggle", "hybrid", "qwen"]

data:
  kaggle:
    raw_dir: "data/raw"
    movielens:
      dataset: "grouplens/movielens-20m-dataset"   # hoặc 25m
      subdir: "movielens"
      ratings_file: "ratings.csv"
      movies_file: "movies.csv"
    movies_metadata:
      dataset: "rounakbanik/the-movies-dataset"
      subdir: "movies_metadata"
      files:
        metadata: "movies_metadata.csv"
        credits: "credits.csv"
        keywords: "keywords.csv"
        links: "links.csv"

  processed:
    base_dir: "data/processed"
    ratings: "data/processed/ratings.parquet"
    movies: "data/processed/movies.parquet"
    movies_metadata: "data/processed/movies_metadata.parquet"
    movie_features: "data/processed/movie_features.parquet"

  split:
    method: "time"
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
    min_ratings_per_user: 5
    min_ratings_per_item: 5

model:
  cf:
    type: "als"             # "als" | "bpr" | "neural_cf"
    factors: 64
    regularization: 0.01
    iterations: 20
    use_gpu: true

  content:
    type: "tfidf"           # hoặc "bert"
    max_features: 50000
    use_overview: true
    use_genres: true
    use_keywords: true
    use_cast_crew: true

  hybrid:
    alpha_cf: 0.7
    alpha_content: 0.3
    use_llm_rerank: true

llm:
  provider: "huggingface"
  model_name: "Qwen/Qwen2.5-1.7B-Instruct"
  max_new_tokens: 128
  temperature: 0.3
  top_p: 0.9
  device: "cuda:0"

  query_parser:
    prompt_template: "config/prompts/query_parser.txt"
  reranker:
    prompt_template: "config/prompts/reranker.txt"
    max_candidates: 50
  explainer:
    prompt_template: "config/prompts/explainer.txt"
    max_explanations: 10

training:
  batch_size: 1024
  num_epochs: 10
  lr: 0.001
  optimizer: "adam"
  num_workers: 4
  checkpoint_dir: "outputs/checkpoints"
  resume_from: null

evaluation:
  metrics:
    - "ndcg@10"
    - "recall@10"
    - "map@10"
  num_negative_samples: 100
  eval_users_sample: 5000

api:
  host: "0.0.0.0"
  port: 8000
  num_workers: 2
  enable_cors: true

experiment:
  name: "baseline_hybrid_qwen_kaggle"
  notes: "Hybrid recsys with Kaggle MovieLens + TMDB and Qwen re-ranking"
  tags:
    - "baseline"
    - "kaggle"
    - "qwen"
```

---

## 8. Gợi ý code rất ngắn cho script Kaggle (dựa trên `kaggle.json`)

```python
# scripts/download_kaggle_datasets.py
import os
import subprocess
import yaml

CONFIG_PATH = "config/config.yml"

def run_cmd(cmd: str):
  print(f"Running: {cmd}")
  subprocess.run(cmd, shell=True, check=True)

def main():
  with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

  kaggle_cfg = cfg["data"]["kaggle"]
  raw_dir = kaggle_cfg["raw_dir"]
  os.makedirs(raw_dir, exist_ok=True)

  for key in ["movielens", "movies_metadata"]:
    d = kaggle_cfg[key]
    dataset = d["dataset"]
    subdir = d["subdir"]
    out_dir = os.path.join(raw_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    cmd = f'kaggle datasets download -d {dataset} -p {out_dir} --unzip'
    run_cmd(cmd)

if __name__ == "__main__":
  main()
```

Kaggle sẽ tự đọc `~/.kaggle/kaggle.json`, nên không cần config gì thêm trong `.env`.

---

Nếu bạn muốn bước tiếp, mình có thể giúp **thiết kế chi tiết `hybrid.py`** (logic CF + content + Qwen), hoặc **prompt cụ thể cho Qwen (query_parser/reranker/explainer)**.
