import re 

class Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())