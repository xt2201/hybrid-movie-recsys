import json
import yaml
import os
from src.llm.qwen_client import QwenClient

CONFIG_PATH = "config/config.yml"

class QueryParser:
    def __init__(self, client: QwenClient = None, config_path: str = CONFIG_PATH):
        if client:
            self.client = client
        else:
            self.client = QwenClient(config_path)
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.prompt_path = self.config["llm"]["query_parser"]["prompt_template"]
        with open(self.prompt_path, "r") as f:
            self.prompt_template = f.read()

    def parse(self, query: str) -> dict:
        prompt = self.prompt_template.format(query=query)
        response = self.client.generate(prompt, system_prompt="You are a helpful assistant that outputs JSON.")
        
        # Clean response (remove markdown code blocks if any)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
            
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {response}")
            return {}

if __name__ == "__main__":
    parser = QueryParser()
    query = "I want to watch a sad sci-fi movie like Interstellar from after 2010."
    result = parser.parse(query)
    print(f"Parsed query: {result}")
