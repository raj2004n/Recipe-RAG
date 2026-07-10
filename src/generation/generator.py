# src/models/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class Generator:
    def __init__(
            self,
            model: str,
            max_input_tokens: int,
            max_new_tokens: int,
            truncation: bool,
            return_tensors: str,
            adapter_path: str | None = None
    ):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.truncation = truncation
        self.return_tensors = return_tensors

        # load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype="auto",
            device_map="auto"
        )

        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, 
                adapter_path
            )
        
    def generate(self, prompt: str) -> str:
        """ Generate recipe response from prompt """
        inputs = self.tokenizer(
            prompt,
            return_tensors=self.return_tensors,
            truncation=self.truncation,
            max_length=self.max_input_tokens
        ).to(self.model.device)  

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )
        
        response = self.tokenizer.batch_decode(outputs)[0]
        
        return response