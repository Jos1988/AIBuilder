from abc import ABC, abstractmethod
import unittest
from typing import List, Optional


class Specification(ABC):

    def __init__(self, name: str, value):
        self.name = name
        self.value = value

    @abstractmethod
    def validate(self):
        pass

    def __call__(self):
        return self.value


class TypeSpecification(Specification):

    def __init__(self, name: str, value, valid_types: List[str]):
        super().__init__(name, value)
        self.valid_types = valid_types

    def validate(self):
        assert self.value in self.valid_types, 'Value {} must be in {}, {} given'\
            .format(self.name, self.valid_types, self.value)


class TestTypeSpecification(unittest.TestCase):

    def test_valid(self):
        self.type_specification = TypeSpecification('test_name', 'test_one', ['test_one', 'test_two'])
        self.type_specification.validate()

    def test_invalid(self):
        self.type_specification = TypeSpecification('test_name', 'test_invalid', ['test_one', 'test_two'])

        with self.assertRaises(AssertionError):
            self.type_specification.validate()


class RangeSpecification(Specification):

    def __init__(self, name: str, value, min_value: int, max_value: int):
        super().__init__(name, value)
        self.min = min_value
        self.max = max_value

    def validate(self):
        assert self.min <= self.value <= self.max, "{} must be in range {} - {}, {} given"\
            .format(self.name, self.min, self.max, self.value)


class TestRangeSpecification(unittest.TestCase):

    def test_valid(self):
        self.range_specification = RangeSpecification('test_name', 5, 1, 10)

        self.range_specification.validate()

    def test_invalid(self):
        self.range_specification = RangeSpecification('test_name', 15, 1, 10)

        with self.assertRaises(AssertionError):
            self.range_specification.validate()


class DataTypeSpecification(Specification):

    def __init__(self, name: str, value, data_type):
        super().__init__(name, value)
        self.data_type = data_type

    def validate(self):
        assert type(self.value) is self.data_type, 'Value \'{}\' must be of type {}, {} given'\
            .format(self.name, self.data_type, type(self.value))


class TestDataTypeSpecification(unittest.TestCase):

    def test_valid(self):
        self.data_type_specification = DataTypeSpecification('test_name', 'test_one', str)
        self.data_type_specification.validate()

        self.data_type_specification = DataTypeSpecification('test_name', ['test_one'], list)
        self.data_type_specification.validate()

        self.data_type_specification = DataTypeSpecification('test_name', {'test': 'one'}, dict)
        self.data_type_specification.validate()

    def test_invalid(self):
        self.data_type_specification = DataTypeSpecification('test_name', 'test_invalid', int)

        with self.assertRaises(AssertionError):
            self.data_type_specification.validate()


class IsCallableSpecification(Specification):
    def validate(self):
        assert callable(self.value), 'Value \'{}\' must be of callable, {} given'\
            .format(self.name, self.value)


class TestIsCallableSpecification(unittest.TestCase):

    def test_valid(self):
        l = lambda x: 1+1

        def fnc():
            return 1+1

        f = fnc

        lmd_specification = IsCallableSpecification('lmd', l)
        lnc_specification = IsCallableSpecification('fnc', f)

        lmd_specification.validate()
        lnc_specification.validate()

    def test_invalid(self):
        lmd_specification = IsCallableSpecification('int', 123)
        lnc_specification = IsCallableSpecification('str', 'test')

        with self.assertRaises(AssertionError):
            lmd_specification.validate()

        with self.assertRaises(AssertionError):
            lnc_specification.validate()


class NullSpecification(Specification):

    def __init__(self, name: str):
        super().__init__(name, None)

    def validate(self):
        assert self.value is None, 'Null Specification not None but, {}.'.format(self.value)


class TestNullSpecification(unittest.TestCase):

    def setUp(self):
        self.specification = NullSpecification('test')

    def test_valid(self):
        self.specification.validate()

    def test_invalid(self):
        self.specification.value = 'something'
        with self.assertRaises(AssertionError):
            self.specification.validate()

    def test_is_none(self):
        self.assertTrue(self.specification() is None)


class Descriptor(Specification):

    def __init__(self, name: str, value: Optional[List]):
        if value is None:
            value = []

        super().__init__(name, value)

    def validate(self):
        pass

    def add_description(self, value: str):
        self.value.append(value)


class TestDescriptor(unittest.TestCase):

    def test_valid(self):
        self.descriptor = Descriptor('description', ['test 1'])

        self.assertEqual(self.descriptor.name, 'description')
        self.assertEqual(self.descriptor.value, ['test 1'])
        self.descriptor.add_description('test 2')
        self.assertEqual(self.descriptor.value, ['test 1', 'test 2'])
        self.descriptor.validate()

    def test_valid_2(self):
        self.descriptor = Descriptor('description', None)

        self.assertEqual(self.descriptor.name, 'description')
        self.assertEqual(self.descriptor.value, [])
        self.descriptor.add_description('test 1')
        self.assertEqual(self.descriptor.value, ['test 1'])
        self.descriptor.validate()


if __name__ == '__main__':
    unittest.main()
