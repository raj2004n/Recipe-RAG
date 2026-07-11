# src/models/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class Generator:
    def __init__(
            self,
            model_id: str,
            max_input_tokens: int,
            max_new_tokens: int,
            truncation: bool,
            return_tensors: str,
            adapter_path: str | None = None
    ):
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.truncation = truncation
        self.return_tensors = return_tensors

        # load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
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
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        genearted_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        
        output_ids = genearted_ids[0][len(model_inputs["input_ids"][0]):].tolist()
        response = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )
        
        return response