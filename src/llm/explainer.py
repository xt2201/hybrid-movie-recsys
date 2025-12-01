import yaml
from src.llm.qwen_client import QwenClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yml"

class Explainer:
    def __init__(self, client: QwenClient = None, config_path: str = CONFIG_PATH):
        if client:
            self.client = client
        else:
            self.client = QwenClient(config_path)
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.prompt_path = self.config["llm"]["explainer"]["prompt_template"]
        with open(self.prompt_path, "r") as f:
            self.prompt_template = f.read()

    def explain(self, query: str, movie: dict) -> str:
        prompt = self.prompt_template.format(
            query=query,
            title=movie['title'],
            overview=movie.get('overview', ''),
            genres=movie.get('genres', '')
        )
        
        response = self.client.generate(prompt, system_prompt="You are a helpful assistant.")
        return response.strip()

if __name__ == "__main__":
    explainer = Explainer()
    query = "sad sci-fi"
    movie = {
        "title": "Interstellar",
        "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "genres": "Adventure|Drama|Sci-Fi"
    }
    explanation = explainer.explain(query, movie)
    logger.info(f"Explanation: {explanation}")
