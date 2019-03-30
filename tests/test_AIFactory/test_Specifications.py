import unittest
from unittest import mock

from AIBuilder.AIFactory.Specifications import TypeSpecification, RangeSpecification, \
    DataTypeSpecification, IsCallableSpecification, NullSpecification, Descriptor, FeatureColumnsSpecification, \
    PrefixedDictSpecification


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

    def test_describe_list(self):
        self.data_type_specification = DataTypeSpecification('test_dir', ['test_one'], list)
        description = self.data_type_specification.describe()
        self.assertEqual(['test_one'], description)

    def test_describe_list_ignore_object(self):
        self.data_type_specification = DataTypeSpecification('test_dir', ['test_one', mock.Mock()], list)
        description = self.data_type_specification.describe()
        self.assertEqual(['test_one'], description)

    def test_describe_list_ignore_callable(self):
        def callMe():
            return 'hello!'

        self.data_type_specification = DataTypeSpecification('test_dir', ['test_one', callMe], list)
        description = self.data_type_specification.describe()
        self.assertEqual(['test_one'], description)

    def test_describe_dict(self):
        self.data_type_specification = DataTypeSpecification('test_dir', {0: 'test', 1: 'test_one'}, dict)
        description = self.data_type_specification.describe()
        self.assertEqual({0: 'test', 1: 'test_one'}, description)

    def test_describe_dict_ignore_object(self):
        self.data_type_specification = DataTypeSpecification('test_dir', {0: 'test', 1: mock.Mock()}, dict)
        description = self.data_type_specification.describe()
        self.assertEqual({0: 'test'}, description)

    def test_describe_dict_ignore_callable(self):
        def callMe():
            return 'hello!'

        self.data_type_specification = DataTypeSpecification('test_dir', {0: 'test', callMe: 1}, dict)
        description = self.data_type_specification.describe()
        self.assertEqual({0: 'test'}, description)


class TestPrefixedDictSpecification(unittest.TestCase):

    def setUp(self) -> None:
        data = {
            'test_dir': 'one',
            'b': 1,
            'c': 0.1,
        }

        self.data_type_specification = PrefixedDictSpecification('test_dir', 'test', data)

    def test_valid(self):
        self.data_type_specification.validate()

    def test_description(self):
        description = self.data_type_specification.describe()
        self.assertEqual('one', description['test_test_dir'])
        self.assertEqual(1, description['test_b'])
        self.assertEqual(0.1, description['test_c'])


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


class TestFeatureColumnsSpecification(unittest.TestCase):
    valid_types = ['type_one', 'type_two']

    def test_valid(self):
        feature_columns = [
            {'name': 'column_name_1', 'type': 'type_one'},
            {'name': 'column_name_2', 'type': 'type_two'}
        ]

        self.specification_setter(feature_columns=feature_columns)
        self.feature_col_specification.validate()

        self.feature_col_specification.add_feature_column('column_name_3', 'type_one')
        self.feature_col_specification.validate()

        values = self.feature_col_specification.value

        self.assertEqual(values[0]['name'], 'column_name_1')
        self.assertEqual(values[0]['type'], 'type_one')

        self.assertEqual(values[1]['name'], 'column_name_2')
        self.assertEqual(values[1]['type'], 'type_two')

        self.assertEqual(values[2]['name'], 'column_name_3')
        self.assertEqual(values[2]['type'], 'type_one')

    def test_invalid(self):
        feature_columns = [
            {'name': 'column_name_1', 'type': 'type_one'},
            {'name': 'column_name_2', 'type': 'invalid'}
        ]
        self.specification_setter(feature_columns=feature_columns)

        with self.assertRaises(AssertionError):
            self.feature_col_specification.validate()

        feature_columns = [
            {'name': 'column_name_1', 'type': 'type_one'},
            {'name': 23, 'type': 'type_two'}
        ]
        self.specification_setter(feature_columns=feature_columns)

        with self.assertRaises(AssertionError):
            self.feature_col_specification.validate()

        feature_columns = [
            {'name': 'column_name', 'type': 'type_one'},
            {'name': 'column_name', 'type': 'type_one'}
        ]
        self.specification_setter(feature_columns=feature_columns)

        with self.assertRaises(AssertionError):
            self.feature_col_specification.validate()

    def test_no_feature_column_set(self):
        feature_columns = []
        self.specification_setter(feature_columns=feature_columns)

        with self.assertRaises(AssertionError):
            self.feature_col_specification.validate()

    def specification_setter(self, feature_columns):
        self.feature_col_specification = FeatureColumnsSpecification('name', feature_columns, self.valid_types)

    def tearDown(self):
        self.feature_col_specification = None


if __name__ == '__main__':
    unittest.main()
