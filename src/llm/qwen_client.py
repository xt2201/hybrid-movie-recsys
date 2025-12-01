import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from typing import List, Dict, Optional
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

CONFIG_PATH = "config/config.yml"

class QwenClient:
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.llm_config = self.config["llm"]
        self.model_name = self.llm_config["model_name"]
        self.device = self.llm_config["device"]
        self.max_new_tokens = self.llm_config["max_new_tokens"]
        self.temperature = self.llm_config["temperature"]
        self.top_p = self.llm_config["top_p"]
        
        logger.info(f"Loading LLM: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()
        logger.info("LLM loaded.")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

if __name__ == "__main__":
    client = QwenClient()
    response = client.generate("Hello, who are you?")
    logger.info(f"Response: {response}")
