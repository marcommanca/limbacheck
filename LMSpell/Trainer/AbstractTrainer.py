from abc import ABC, abstractmethod
import pandas as pd

class AbstractTrainer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass