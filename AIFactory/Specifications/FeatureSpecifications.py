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
