# src/models/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class Generator:
    def __init__(self, config: dict):
        model_name = config["generator"]["model_name"]
        adapter_path = config["generator"]["adapter_path"]
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, 
                adapter_path
            )
        
    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """ Generate recipe response from prompt """
        max_new_tokens = 512 # hard coded for now

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)  

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # extract only the answer part after "ANSWER:"
        if "ANSWER:" in response:
            response = response.split("ANSWER:")[-1].strip()
        
        return response