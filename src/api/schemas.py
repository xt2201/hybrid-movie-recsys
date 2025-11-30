from pydantic import BaseModel
from typing import List, Optional, Any

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    query: Optional[str] = None
    num_items: int = 10
    use_llm: bool = True

class MovieResponse(BaseModel):
    id: int
    title: str
    genres: Optional[str] = None
    overview: Optional[str] = None
    score: float
    explanation: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: Optional[int]
    query: Optional[str]
    recommendations: List[MovieResponse]
    parsed_preferences: Optional[dict] = None
