import json
import yaml
from typing import List, Dict
from src.llm.qwen_client import QwenClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yml"

class Reranker:
    def __init__(self, client: QwenClient = None, config_path: str = CONFIG_PATH):
        if client:
            self.client = client
        else:
            self.client = QwenClient(config_path)
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.prompt_path = self.config["llm"]["reranker"]["prompt_template"]
        with open(self.prompt_path, "r") as f:
            self.prompt_template = f.read()

    def rerank(self, query: str, preferences: dict, candidates: List[Dict]) -> List[Dict]:
        # candidates: list of dicts with 'id', 'title', 'overview', 'genres'
        candidates_str = ""
        for c in candidates:
            candidates_str += f"- ID: {c['id']}, Title: {c['title']}, Genres: {c['genres']}\n"
            
        prompt = self.prompt_template.format(
            query=query,
            preferences=json.dumps(preferences),
            candidates=candidates_str
        )
        
        response = self.client.generate(prompt, system_prompt="You are a helpful assistant that outputs JSON.")
        
        # Clean response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
            
        try:
            ranked_list = json.loads(response)
            # Sort by score descending
            ranked_list.sort(key=lambda x: x.get('score', 0), reverse=True)
            return ranked_list
        except json.JSONDecodeError:
            logger.info(f"Failed to parse JSON: {response}")
            return []

if __name__ == "__main__":
    reranker = Reranker()
    query = "sad sci-fi"
    prefs = {"mood": "sad", "must_genres": ["Sci-Fi"]}
    candidates = [
        {"id": 1, "title": "Interstellar", "genres": "Adventure|Drama|Sci-Fi"},
        {"id": 2, "title": "The Martian", "genres": "Adventure|Drama|Sci-Fi"},
        {"id": 3, "title": "Guardians of the Galaxy", "genres": "Action|Sci-Fi|Comedy"}
    ]
    result = reranker.rerank(query, prefs, candidates)
    logger.info(f"Reranked: {result}")
