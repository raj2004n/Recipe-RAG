from abc import ABC, abstractmethod
import json
import os

class BaseStrategy(ABC):
    @classmethod
    def set_strategy(cls, strategy_name: str, metadata_path: str) -> None:
        # import inside the method to avoid circular imports when dense inherits base
        # downside, strategies dict duplicated in set_strategy, from_metadata
        from .dense import DenseStrategy

        strategies = {
            "dense": DenseStrategy,
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # save strategy into metadata
        if metadata_path is None:
            os.makedirs(metadata_path, exist_ok=True)
            metadata_path = os.path.join("snapshots/building", "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "strategy": strategy
                }, f, indent=2)
        else:
            os.makedirs(metadata_path, exist_ok=True)
            metadata_path = os.path.join(metadata_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "strategy": strategy_name
                }, f, indent=2)


    @classmethod
    def from_metadata(cls, metadata_path: str) -> "BaseStrategy":
        # import inside the method to avoid circular imports when dense inherits base
        # downside, strategies dict duplicated in set_strategy, from_metadata
        from .dense import DenseStrategy

        strategies = {
            "dense": DenseStrategy,
        }

        metadata_path = os.path.join(metadata_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        strategy_name = metadata["strategy"]
    
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategies[strategy_name]()


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


    @abstractmethod
    def finalise_build(self):
        pass