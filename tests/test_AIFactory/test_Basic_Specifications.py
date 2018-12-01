import unittest
from AIBuilder.AIFactory.Specifications.BasicSpecifications import TypeSpecification, RangeSpecification, \
    DataTypeSpecification, IsCallableSpecification, NullSpecification, Descriptor


class TestTypeSpecification(unittest.TestCase):

    def test_valid(self):
        self.type_specification = TypeSpecification('test_dir', 'test_one', ['test_one', 'test_two'])
        self.type_specification.validate()

    def test_invalid(self):
        self.type_specification = TypeSpecification('test_dir', 'test_invalid', ['test_one', 'test_two'])

        with self.assertRaises(AssertionError):
            self.type_specification.validate()


class TestRangeSpecification(unittest.TestCase):

    def test_valid(self):
        self.range_specification = RangeSpecification('test_dir', 5, 1, 10)

        self.range_specification.validate()

    def test_invalid(self):
        self.range_specification = RangeSpecification('test_dir', 15, 1, 10)

        with self.assertRaises(AssertionError):
            self.range_specification.validate()


class TestDataTypeSpecification(unittest.TestCase):

    def test_valid(self):
        self.data_type_specification = DataTypeSpecification('test_dir', 'test_one', str)
        self.data_type_specification.validate()

        self.data_type_specification = DataTypeSpecification('test_dir', ['test_one'], list)
        self.data_type_specification.validate()

        self.data_type_specification = DataTypeSpecification('test_dir', {'test_dir': 'one'}, dict)
        self.data_type_specification.validate()

    def test_invalid(self):
        self.data_type_specification = DataTypeSpecification('test_dir', 'test_invalid', int)

        with self.assertRaises(AssertionError):
            self.data_type_specification.validate()


class TestIsCallableSpecification(unittest.TestCase):

    def test_valid(self):
        l = lambda x: 1 + 1

        def fnc():
            return 1 + 1

        f = fnc

        lmd_specification = IsCallableSpecification('lmd', l)
        lnc_specification = IsCallableSpecification('fnc', f)

        lmd_specification.validate()
        lnc_specification.validate()

    def test_invalid(self):
        lmd_specification = IsCallableSpecification('int', 123)
        lnc_specification = IsCallableSpecification('str', 'test_dir')

        with self.assertRaises(AssertionError):
            lmd_specification.validate()

        with self.assertRaises(AssertionError):
            lnc_specification.validate()


class TestNullSpecification(unittest.TestCase):

    def setUp(self):
        self.specification = NullSpecification('test_dir')

    def test_valid(self):
        self.specification.validate()

    def test_invalid(self):
        self.specification.value = 'something'
        with self.assertRaises(AssertionError):
            self.specification.validate()

    def test_is_none(self):
        self.assertTrue(self.specification() is None)


class TestDescriptor(unittest.TestCase):

    def test_valid(self):
        self.descriptor = Descriptor('description', ['test_dir 1'])

        self.assertEqual(self.descriptor.name, 'description')
        self.assertEqual(self.descriptor.value, ['test_dir 1'])
        self.descriptor.add_description('test_dir 2')
        self.assertEqual(self.descriptor.value, ['test_dir 1', 'test_dir 2'])
        self.descriptor.validate()

    def test_valid_2(self):
        self.descriptor = Descriptor('description', None)

        self.assertEqual(self.descriptor.name, 'description')
        self.assertEqual(self.descriptor.value, [])
        self.descriptor.add_description('test_dir 1')
        self.assertEqual(self.descriptor.value, ['test_dir 1'])
        self.descriptor.validate()


if __name__ == '__main__':
    unittest.main()
