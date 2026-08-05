from abc import ABC, abstractmethod

class BaseStrategy(ABC):
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