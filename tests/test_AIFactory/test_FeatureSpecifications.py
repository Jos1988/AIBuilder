import unittest
from AIBuilder.AIFactory.Specifications.FeatureSpecifications import FeatureColumnsSpecification


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
