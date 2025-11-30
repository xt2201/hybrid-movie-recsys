import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.schemas import RecommendationRequest, RecommendationResponse, MovieResponse
from src.recommender.hybrid import HybridRecommender
from src.llm.query_parser import QueryParser
from src.llm.reranker import Reranker
from src.llm.explainer import Explainer
from src.llm.qwen_client import QwenClient

CONFIG_PATH = "config/config.yml"

# Global variables for models
recsys = None
qwen_client = None
query_parser = None
reranker = None
explainer = None
movie_data_map = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recsys, qwen_client, query_parser, reranker, explainer, movie_data_map
    
    print("Loading models...")
    # Load Recommender
    recsys = HybridRecommender(CONFIG_PATH)
    # For demo, we fit on startup. In prod, load from checkpoint.
    # To save time, we might want to load a saved model if available.
    # For now, let's fit (it takes a few seconds with SVD).
    recsys.fit()
    
    # Create a map for quick movie lookup
    movies_df = recsys.dataset.movies
    # movies_df has 'movieId', 'title', 'genres', 'overview'
    # We need to map movieId (internal?) to details.
    # In dataset.py, item_map maps internal index to original movieId?
    # No, item_map maps original movieId -> internal index.
    # But movies_df has 'movieId' which is original ID.
    
    # Let's create a map: internal_idx -> movie_row
    for _, row in movies_df.iterrows():
        mid = row['movieId']
        if mid in recsys.dataset.item_map:
            idx = recsys.dataset.item_map[mid]
            movie_data_map[idx] = row.to_dict()
            
    # Load LLM components
    # Only load if configured
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    if config["model"]["hybrid"]["use_llm_rerank"]:
        qwen_client = QwenClient(CONFIG_PATH)
        query_parser = QueryParser(qwen_client, CONFIG_PATH)
        reranker = Reranker(qwen_client, CONFIG_PATH)
        explainer = Explainer(qwen_client, CONFIG_PATH)
        
    print("Models loaded.")
    yield
    print("Shutting down...")

app = FastAPI(title="Hybrid Movie Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    if recsys is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    recommendations = []
    parsed_prefs = {}
    
    # 1. Parse Query if present
    if request.query and query_parser:
        parsed_prefs = query_parser.parse(request.query)
        # TODO: Use parsed prefs to filter or boost initial candidates
        # For now, we just pass it to reranker
        
    # 2. Get Candidates
    # If user_id is provided, use it. If not, use a default or cold-start strategy.
    # If only query is provided, we might need a search module (Content-based search).
    # Current HybridRecommender requires user_id.
    
    # Fallback: if no user_id, use user 1 (or handle cold start)
    user_id = request.user_id if request.user_id is not None else 1
    
    # Get more candidates for re-ranking
    raw_recs = recsys.recommend(user_id, N=request.num_items * 2)
    
    # Convert to candidate list for LLM
    candidates = []
    for idx, score in raw_recs:
        if idx in movie_data_map:
            m = movie_data_map[idx]
            candidates.append({
                "id": idx, # Internal ID
                "title": m.get('title', 'Unknown'),
                "genres": m.get('genres', ''),
                "overview": m.get('overview', ''),
                "base_score": score
            })
            
    # 3. LLM Re-ranking
    if request.use_llm and request.query and reranker and candidates:
        ranked_results = reranker.rerank(request.query, parsed_prefs, candidates)
        
        # Merge results
        # ranked_results has 'id', 'score', 'reason'
        # We need to map back to full movie details
        
        final_recs = []
        for r in ranked_results:
            rid = r.get('id')
            # Find original candidate
            cand = next((c for c in candidates if c['id'] == rid), None)
            if cand:
                final_recs.append(MovieResponse(
                    id=cand['id'],
                    title=cand['title'],
                    genres=cand['genres'],
                    overview=cand['overview'],
                    score=float(r.get('score', 0)),
                    explanation=r.get('reason')
                ))
                
        # Fill with remaining candidates if not enough
        if len(final_recs) < request.num_items:
            existing_ids = {r.id for r in final_recs}
            for c in candidates:
                if c['id'] not in existing_ids:
                    final_recs.append(MovieResponse(
                        id=c['id'],
                        title=c['title'],
                        genres=c['genres'],
                        overview=c['overview'],
                        score=c['base_score'],
                        explanation="Recommended based on your history."
                    ))
                    if len(final_recs) >= request.num_items:
                        break
                        
        recommendations = final_recs[:request.num_items]
        
    else:
        # No LLM or no query, just return baseline recs
        for c in candidates[:request.num_items]:
            recommendations.append(MovieResponse(
                id=c['id'],
                title=c['title'],
                genres=c['genres'],
                overview=c['overview'],
                score=c['base_score'],
                explanation="Recommended based on your history."
            ))

    # 4. Generate Explanations (if enabled and not already generated by reranker)
    # Reranker already generates 'reason'.
    
    return RecommendationResponse(
        user_id=user_id,
        query=request.query,
        recommendations=recommendations,
        parsed_preferences=parsed_prefs
    )

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
