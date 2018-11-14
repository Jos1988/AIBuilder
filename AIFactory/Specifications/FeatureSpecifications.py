import unittest
from typing import List
from AIBuilder.AIFactory.Specifications.BasicSpecifications import Specification


class FeatureColumnsSpecification(Specification):

    def __init__(self, name: str, value: List[dict], valid_types: List[str]):
        super().__init__(name, value)
        self.valid_types = valid_types

    def validate(self):
        column_names = []
        assert len(self.value) != 0, 'no feature columns set.'

        for feature_column in self.value:
            assert type(feature_column['name']) is str, 'feature column name must be str, {} given'\
                .format(feature_column['name'])

            assert feature_column['name'] not in column_names, 'feature column {} already in column list, {}'.\
                format(feature_column['name'], self.value)
            column_names.append(feature_column['name'])

            assert feature_column['type'] in self.valid_types, 'Value {} must be in {}, {} given'\
                .format(self.name, self.valid_types, self.value)

    def add_feature_column(self, name: str, column_type: str):
        new_column_data = {'name': name, 'type': column_type}
        self.value.append(new_column_data)


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
