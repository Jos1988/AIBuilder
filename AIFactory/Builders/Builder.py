from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI


class Builder(ABC):
    # add you new builder types here.
    ESTIMATOR = 'estimator'
    OPTIMIZER = 'optimizer'
    DATA_MODEL = 'data_model'
    SCRUBBER = 'scrubber'

    @property
    @abstractmethod
    def dependent_on(self) -> list:
        pass

    @property
    @abstractmethod
    def builder_type(self) -> str:
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def build(self, neural_net: AbstractAI):
        pass
