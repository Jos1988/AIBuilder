from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Specifications.BasicSpecifications import Specification
from unittest import TestCase, mock


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


class TestBuilder(Builder):

    @property
    def dependent_on(self) -> list:
        pass

    @property
    def builder_type(self) -> str:
        pass

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        pass


class BuilderTest(TestCase):

    def setUp(self):
        self.builder = TestBuilder()
        self.specification_one = mock.patch('Builder.Specification')
        self.specification_one.name = 'spec1'
        self.specification_one.value = 'value1'
        self.specification_one.validate = mock.Mock()

        self.builder.get_specs = mock.Mock()
        self.builder.get_specs.return_value = [self.specification_one]

    def test_describe(self):
        description = self.builder.describe()
        self.assertEqual({'spec1': 'value1'}, description)

    def test_validate(self):
        self.builder.validate()
        self.specification_one.validate.assert_called_once()
