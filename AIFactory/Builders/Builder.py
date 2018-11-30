from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Specifications.BasicSpecifications import Specification
# todo: move files to one file for easier importing


class Builder(ABC):
    # add you new builder types here.
    ESTIMATOR = 'estimator'
    OPTIMIZER = 'optimizer'
    DATA_MODEL = 'data_model'
    SCRUBBER = 'scrubber'
    INPUT_FUNCTION = 'input_function'
    NAMING_SCHEME = 'naming'

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

    def describe(self):
        description = {}
        specs = self.get_specs()
        for spec in specs:
            description[spec.name] = spec.value

        return description

    def validate_specifications(self):
        specs = self.get_specs()

        for spec in specs:
            spec.validate()

    def get_specs(self):
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), Specification)]

    @abstractmethod
    def build(self, neural_net: AbstractAI):
        pass
