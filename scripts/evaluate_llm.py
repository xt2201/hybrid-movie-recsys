import yaml
import yaml
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.qwen_client import QwenClient
from src.llm.query_parser import QueryParser
from src.llm.reranker import Reranker
from src.llm.explainer import Explainer

CONFIG_PATH = "config/config.yml"

def main():
    print("Initializing Qwen Client...")
    client = QwenClient(CONFIG_PATH)
    
    print("\n=== Testing Query Parser ===")
    parser = QueryParser(client, CONFIG_PATH)
    query = "I want to watch a sad sci-fi movie like Interstellar from after 2010."
    print(f"Query: {query}")
    parsed = parser.parse(query)
    print(f"Parsed: {parsed}")
    
    print("\n=== Testing Reranker ===")
    reranker = Reranker(client, CONFIG_PATH)
    candidates = [
        {"id": 1, "title": "Interstellar", "genres": "Adventure|Drama|Sci-Fi", "overview": "A team of explorers travel through a wormhole in space."},
        {"id": 2, "title": "The Martian", "genres": "Adventure|Drama|Sci-Fi", "overview": "An astronaut becomes stranded on Mars after his team assumes him dead."},
        {"id": 3, "title": "Guardians of the Galaxy", "genres": "Action|Sci-Fi|Comedy", "overview": "A group of intergalactic criminals must pull together to stop a fanatical warrior."},
        {"id": 4, "title": "La La Land", "genres": "Comedy|Drama|Music", "overview": "A jazz pianist falls for an aspiring actress in Los Angeles."}
    ]
    print(f"Candidates: {[c['title'] for c in candidates]}")
    ranked = reranker.rerank(query, parsed, candidates)
    print("Ranked Results:")
    for item in ranked:
        print(f"- {item.get('title', 'Unknown')} (Score: {item.get('score')}) - {item.get('reason')}")
        
    print("\n=== Testing Explainer ===")
    explainer = Explainer(client, CONFIG_PATH)
    if ranked:
        top_movie_id = ranked[0]['id']
        # Find movie dict
        top_movie = next(c for c in candidates if c['id'] == top_movie_id)
        explanation = explainer.explain(query, top_movie)
        print(f"Explanation for {top_movie['title']}: {explanation}")

if __name__ == "__main__":
    main()
