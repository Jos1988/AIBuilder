from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Specifications.specification import Specification


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

    def validate_specifications(self):
        specs = [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), Specification)]

        for spec in specs:
            spec.validate()

    @abstractmethod
    def build(self, neural_net: AbstractAI):
        pass
