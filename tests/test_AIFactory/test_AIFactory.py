from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.Data import MetaData
import unittest
from unittest import mock, TestCase
from AIBuilder.AIFactory import AIFactory


class TestAIFactory(TestCase):

    def setUp(self):
        self.factory = AIFactory.AIFactory(project_name='test', log_dir='test/dir')

    def test_create_AI(self):
        metadata = MetaData()
        metadata.define_numerical_columns(['feature_2', 'feature_3', 'target_1'])
        metadata.define_categorical_columns(['feature_1'])

        builder_A = mock.Mock(spec=Builder)
        builder_A.dependent_on = ['B']
        builder_A.builder_type = 'A'
        builder_A.describe = mock.Mock()
        builder_A.describe.return_value = {'A': 1}
        builder_A.build = mock.Mock()

        builder_B = mock.Mock(spec=Builder)
        builder_B.dependent_on = ['C']
        builder_B.builder_type = 'B'
        builder_B.describe = mock.Mock()
        builder_B.describe.return_value = {'B': 1}
        builder_B.build = mock.Mock()

        builder_C = mock.Mock(spec=Builder)
        builder_C.dependent_on = []
        builder_C.builder_type = 'C'
        builder_C.describe = mock.Mock()
        builder_C.describe.return_value = {'C': 1}
        builder_C.build = mock.Mock()

        artie = self.factory.create_AI([builder_A, builder_B, builder_C])
        self.assertEqual(artie.description, {'C': {'C': 1}, 'B': {'B': 1}, 'A': {'A': 1}})
        self.assertEqual(self.factory.builders_sorted, [builder_C, builder_B, builder_A])

        builder_A.build.assert_called_once_with(artie)
        builder_B.build.assert_called_once_with(artie)
        builder_C.build.assert_called_once_with(artie)


if __name__ == '__main__':
    unittest.main()
