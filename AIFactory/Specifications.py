from abc import ABC, abstractmethod
from typing import List, Optional
import tensorflow as tf


class Describer:
    """ Can describe itself by loading its spec descriptions into self.description (dict). """

    def describe(self):
        description = {}
        specs = self.get_specs()
        for spec in specs:
            description[spec.name] = spec.describe()

        return description

    def validate_specifications(self):
        specs = self.get_specs()

        for spec in specs:
            spec.validate()

    def get_specs(self):
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), Specification)]


class Specification(ABC):
    """ Can be wrapped around a class value to include it in the class description. """

    def __init__(self, name: str, value):
        self.name = name
        self.value = value

    @abstractmethod
    def validate(self):
        pass

    def describe(self):
        return self.value

    def __call__(self):
        return self.value


class TypeSpecification(Specification):

    def __init__(self, name: str, value, valid_types: List[str]):
        super().__init__(name, value)
        self.valid_types = valid_types

    def validate(self):
        assert self.value in self.valid_types, 'Value {} must be in {}, {} given' \
            .format(self.name, self.valid_types, self.value)


class RangeSpecification(Specification):

    def __init__(self, name: str, value, min_value: int, max_value: int):
        super().__init__(name, value)
        self.min = min_value
        self.max = max_value

    def validate(self):
        assert self.min <= self.value <= self.max, "{} must be in range {} - {}, {} given" \
            .format(self.name, self.min, self.max, self.value)


def is_primitive(var):
    """ todo: move logic on if and how different types are described to pattern and hook up to describer instead of
            specification. """
    return isinstance(var, (int, float, bool, str, dict, list))


class DataTypeSpecification(Specification):

    def __init__(self, name: str, value, data_type):
        super().__init__(name, value)
        self.data_type = data_type

    def describe(self):
        description = self.value

        if self.data_type is dict:
            description = {}
            for label, value in self.value.items():
                # todo: check on abstract class in isinstance and use multiple inheritance to implement is.
                if is_primitive(label) and (is_primitive(value) or isinstance(value, ConfigDescriber)):
                    description[label] = value

        if self.data_type is list:
            description = []
            for item in self.value:
                if is_primitive(item):
                    description.append(item)

        return description

    def validate(self):
        assert type(self.value) is self.data_type, 'Value \'{}\' must be of type {}, {} given' \
            .format(self.name, self.data_type, type(self.value))


class PrefixedDictSpecification(DataTypeSpecification):
    """ Saves dict of data, but prefixes the keys.

    data = {'snake': 'constrictor', 'cheese': 'Brie'}
    Example: PrefixedDictSpecification('name', 'ham', data)

    description:
    'name': {'ham_snake': 'constrictor', 'ham_cheese': 'Brie'}

    """
    def __init__(self, name: str, prefix: str, value):

        super().__init__(name, value, dict)
        self.prefix = prefix

    def describe(self):
        description = super().describe()
        prefixed_description = {}
        for label, value in description.items():
            prefixed_label = self.prefix + '_' + label
            prefixed_description[prefixed_label] = value

        return prefixed_description


class IsCallableSpecification(Specification):
    def validate(self):
        assert callable(self.value), 'Value \'{}\' must be of callable, {} given' \
            .format(self.name, self.value)


class NullSpecification(Specification):

    def __init__(self, name: str):
        super().__init__(name, None)

    def validate(self):
        assert self.value is None, 'Null Specification not None but, {}.'.format(self.value)


# todo: temp solution for scrubbers.
class Descriptor(Specification):

    def __init__(self, name: str, value: Optional[List]):
        if value is None:
            value = []

        super().__init__(name, value)

    def validate(self):
        pass

    def add_description(self, value: str):
        self.value.append(value)


class FeatureColumnsSpecification(Specification):

    def __init__(self, name: str, value: List[dict], valid_types: List[str]):
        super().__init__(name, value)
        self.valid_types = valid_types

    def validate(self):
        column_names = []
        assert len(self.value) != 0, 'no feature columns set.'

        for feature_column in self.value:
            assert type(feature_column['name']) is str, 'feature column name must be str, {} given' \
                .format(feature_column['name'])

            assert feature_column['name'] not in column_names, 'feature column {} already in column list, {}'. \
                format(feature_column['name'], self.value)
            column_names.append(feature_column['name'])

            assert feature_column['type'] in self.valid_types, 'Value {} must be in {}, {} given' \
                .format(self.name, self.valid_types, self.value)

    def add_feature_column(self, name: str, column_type: str):
        new_column_data = {'name': name, 'type': column_type}
        self.value.append(new_column_data)


# todo: move implementation from abstraction module?
class ConfigDescriber(tf.estimator.RunConfig):
    """ Wrapper around estimator config so we can describe it. """
    description: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return str(self.description)

    def __repr__(self):
        return str(self.description)
