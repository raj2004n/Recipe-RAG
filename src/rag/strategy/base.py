from abc import ABC, abstractmethod
import json
import os

class BaseStrategy(ABC):
    @classmethod
    def from_metadata(cls, metadata_path: str) -> "BaseStrategy":
        # import inside the method to avoid circular imports when dense inherits base
        # but this means that strategies dict is duplicated
        from .dense import DenseStrategy

        strategies = {
            "dense": DenseStrategy,
        }

        metadata_path = os.path.join(metadata_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        name = metadata["strategy"]
    
        if name not in strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        return strategies[name]()


    @abstractmethod
    def chunk(self):
        pass 


    @abstractmethod
    def transform(self, chunk_sets):
        pass
    # calling this transform to respect the naming conventions 
    # dense / sparse retriever encodes / tokenizes, so this 
    # naming allows the main intent to be carried through 
    # while being flexible

    @abstractmethod
    def index(self):
        pass


    #@abstractmethod
    #def finalise_build(self):
        #pass