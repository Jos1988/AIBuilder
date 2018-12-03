from unittest import mock
from AIBuilder.AI import AI, AbstractAI
from AIBuilder.Data import MetaData, DataModel
from AIBuilder.AIFactory.Specifications import TypeSpecification
import unittest
import numpy
import pandas as pd
from AIBuilder.AIFactory.Builders import Builder
import AIBuilder.DataScrubbing as scrubber
from AIBuilder.AIFactory.Builders import DataBuilder, EstimatorBuilder, InputFunctionBuilder, NamingSchemeBuilder, \
    OptimizerBuilder, ScrubAdapter, MetadataBuilder


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


class BuilderTest(unittest.TestCase):

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


class TestDataBuilder(unittest.TestCase):

    def test_build(self):
        data_builder = DataBuilder(data_source='C:/python/practice2/AIBuilder/tests/data/test_data.csv',
                                   target_column='target_1',
                                   validation_data_percentage=20,
                                   feature_columns={},
                                   data_columns=[],
                                   metadata=MetaData())

        data_builder.add_feature_column(name='feature_1', column_type=DataBuilder.CATEGORICAL_COLUMN_VOC_LIST)
        data_builder.add_feature_column(name='feature_2', column_type=DataBuilder.NUMERICAL_COLUMN)
        data_builder.add_feature_column(name='feature_3', column_type=DataBuilder.NUMERICAL_COLUMN)

        arti = AI(project_name='name', log_dir='path/to/dir')
        data_builder.validate()
        data_builder.build(ai=arti)

        feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.validate_data_frame(arti.training_data, feature_names)
        self.validate_data_frame(arti.evaluation_data, feature_names)

    def test_constructor_build(self):
        data_builder = DataBuilder(data_source='C:/python/practice2/AIBuilder/tests/data/test_data.csv',
                                   target_column='target_1',
                                   validation_data_percentage=20,
                                   feature_columns={
                                       'feature_1': DataBuilder.CATEGORICAL_COLUMN_VOC_LIST,
                                       'feature_2': DataBuilder.NUMERICAL_COLUMN,
                                       'feature_3': DataBuilder.NUMERICAL_COLUMN
                                   },
                                   data_columns=[],
                                   metadata=MetaData())

        arti = AI(project_name='name', log_dir='path/to/dir')
        data_builder.validate()
        data_builder.build(ai=arti)

        feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.validate_data_frame(arti.training_data, feature_names)
        self.validate_data_frame(arti.evaluation_data, feature_names)

    def validate_data_frame(self, data_frame: DataModel, feature_name_list: list):
        self.assertEqual(data_frame.feature_columns_names, feature_name_list)
        self.assertEqual(data_frame.target_column_name, 'target_1')

        for tf_feature_column in data_frame.get_tf_feature_columns():
            self.assertTrue(tf_feature_column.name in feature_name_list)


class TestEstimatorBuilder(unittest.TestCase):

    def test_validate(self):
        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        estimator_builder.validate()

    def test_invalid_estimator_type(self):
        invalid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        invalid_estimator_builder.estimator_type = TypeSpecification(name=EstimatorBuilder.ESTIMATOR,
                                                                     value='invalid',
                                                                     valid_types=EstimatorBuilder.valid_estimator_types)

        valid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)

        with self.assertRaises(AssertionError):
            invalid_estimator_builder.validate()

        with self.assertRaises(AssertionError):
            valid_estimator_builder.set_estimator('invalid')
            valid_estimator_builder.validate()

        with self.assertRaises(AssertionError):
            builder = EstimatorBuilder('invalid')
            builder.validate()

    def test_build(self):
        mock_data_model = mock.MagicMock()
        mock_optimizer = mock.MagicMock()

        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)

        mock_data_model.get_tf_feature_columns.return_value = []

        arti = mock.Mock('EstimatorBuilder.AbstractAI')
        arti.get_log_dir = mock.Mock()
        arti.get_log_dir.return_value = 'path/to/log'
        arti.get_project_name = mock.Mock()
        arti.get_project_name.return_value = 'project_name'
        arti.get_name = mock.Mock()
        arti.get_name.return_value = 'ai_name'
        arti.set_estimator = mock.Mock()
        arti.optimizer = mock_optimizer
        arti.training_data = mock_data_model

        estimator_builder.build(arti)
        arti.set_estimator.assert_called_once()


class TestInputFnBuilder(unittest.TestCase):

    def setUp(self):
        self.input_fn_builder = InputFunctionBuilder(train_fn=InputFunctionBuilder.BASE_FN,
                                                     train_kwargs={'batch_size': 100, 'epoch': 1},
                                                     evaluation_fn=InputFunctionBuilder.BASE_FN,
                                                     evaluation_kwargs={'batch_size': 100, 'epoch': 1})

    def test_validate(self):
        self.input_fn_builder.validate()

    def test_build(self):
        train_data = mock.Mock()
        eval_data = mock.Mock()

        arti = mock.Mock()
        arti.training_data = train_data
        arti.evaluation_data = eval_data

        arti.set_training_fn = mock.Mock()
        arti.set_evaluation_fn = mock.Mock()

        self.input_fn_builder.build(arti)

        arti.set_training_fn.assert_called_once()
        arti.set_evaluation_fn.assert_called_once()


class TestNamingScheme(unittest.TestCase):

    def setUp(self):
        self.naming_scheme = NamingSchemeBuilder()
        self.arti = mock.Mock('test_Builders.NamingSchemeBuilder.AIBuilder.AI')
        self.arti.get_log_dir = mock.Mock()
        self.arti.get_log_dir.return_value = '../../../builder projects/log'
        self.arti.get_project_name = mock.Mock()
        self.arti.set_name = mock.Mock()
        self.arti.get_name = mock.Mock()
        self.arti.get_name.return_value = None

    @mock.patch('AIBuilder.AIFactory.Builders.os.walk')
    def test_generate_name(self, walk):
        walk.return_value = iter([[None, ['shoesies', 'shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoes'
        self.naming_scheme.build(self.arti)
        self.arti.set_name.assert_called_once_with('shoes_3')

    @mock.patch('AIBuilder.AIFactory.Builders.os.walk')
    def test_numerate_name(self, walk):
        walk.return_value = iter([[None, ['shoesies', 'shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoesies'
        self.naming_scheme.build(self.arti)
        self.arti.set_name.assert_called_once_with('shoesies_1')


class TestOptimizerBuilder(unittest.TestCase):

    def test_valid_validate(self):
        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=1.0)

        optimizer_builder_no_clipping.validate()
        optimizer_builder_with_clipping.validate()

    def test_invalid_validate(self):
        optimizer_builder = OptimizerBuilder(
            optimizer_type='invalid',
            learning_rate=5.0,
            gradient_clipping=0.0002)

        with self.assertRaises(AssertionError):
            optimizer_builder.validate()

    @mock.patch('AIBuilder.AIFactory.Builders.tf.contrib.estimator.clip_gradients_by_norm')
    @mock.patch('AIBuilder.AIFactory.Builders.tf.train.GradientDescentOptimizer')
    def test_build_with_clipping(self, mock_optimizer, mock_clipper):
        arti = mock.Mock('OptimizerBuilder.AbstractAI')
        arti.set_optimizer = mock.Mock()

        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=0.1)

        optimizer_builder_with_clipping.build(arti)
        arti.set_optimizer.assert_called_once()
        mock_optimizer.assert_called_with(learning_rate=1.0)
        mock_clipper.assert_called()

    @mock.patch('AIBuilder.AIFactory.Builders.tf.train.GradientDescentOptimizer')
    def test_build_with_no_clipping(self, mock_optimizer):
        arti = mock.Mock('OptimizerBuilder.AbstractAI')
        arti.set_optimizer = mock.Mock()

        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_no_clipping.build(arti)
        arti.set_optimizer.assert_called_once()
        mock_optimizer.assert_called_with(learning_rate=1.0)


class TestScrubAdapter(unittest.TestCase):

    def setUp(self):
        self.scrubber_three = mock.Mock('ScrubAdapter.scrubber')
        self.scrubber_four = mock.Mock('ScrubAdapter.scrubber')
        self.scrubber_one = mock.Mock(spec=scrubber.Scrubber)
        self.scrubber_two = mock.Mock(spec=scrubber.Scrubber)
        self.scrub_adapter = ScrubAdapter([self.scrubber_one, self.scrubber_two])
        self.scrub_adapter.add_scrubber(self.scrubber_three)
        self.scrub_adapter.add_scrubber(self.scrubber_four)

    def test_add_scrubber(self):
        self.assertIn(self.scrubber_one, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_two, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_three, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_four, self.scrub_adapter.and_scrubber.scrubber_list)

    def test_build(self):
        arti = mock.Mock('ScrubAdapter.AbstractAI')
        training_data = mock.Mock('ScrubAdapter.Data.DataModel')
        training_data.metadata = mock.Mock(name='training_metadata')

        evaluation_data = mock.patch('ScrubAdapter.Data.DataModel')
        evaluation_data.metadata = mock.Mock(name='validation_metadata')

        and_scrubber = mock.Mock(name='and_scrubber')
        and_scrubber.validate_metadata = mock.Mock(name='and_scrubber_validate_metadata')
        and_scrubber.scrub = mock.Mock(name='and_scrubber_scrub')

        arti.training_data = training_data
        arti.evaluation_data = evaluation_data
        self.scrub_adapter.and_scrubber = and_scrubber

        self.scrub_adapter.build(arti)

        and_scrubber.validate_metadata.assert_any_call(training_data.metadata),
        and_scrubber.scrub.assert_any_call(training_data),
        and_scrubber.validate_metadata.assert_any_call(evaluation_data.metadata),
        and_scrubber.scrub.assert_any_call(evaluation_data)


class TestMetadataBuilder(unittest.TestCase):

    def test_build(self):
        builder = MetadataBuilder()
        arti = mock.Mock('AIBuilder.AbstractAI')

        # mock evaluation model
        data = {'col1': ['cat1', 'cat2'], 'col2': [3, 4], 'col3': [0.1, 0.2], 'col4': [True, False]}
        dataframe = pd.DataFrame(data=data)
        evaluation_model = DataModel(dataframe)

        arti.get_evaluation_data = mock.Mock()
        arti.get_evaluation_data.return_value = evaluation_model

        # mock training model
        data = {'col1': ['cat1', 'cat2'], 'col2': [3, 4], 'col3': [0.1, 0.2], 'col4': [True, False]}
        dataframe = pd.DataFrame(data=data)
        training_model = DataModel(dataframe)

        arti.get_training_data = mock.Mock()
        arti.get_training_data.return_value = training_model

        # build
        builder.build(arti)

        # assert
        self.assertListEqual(training_model.metadata.categorical_columns, ['col1', 'col4'])
        self.assertListEqual(training_model.metadata.numerical_columns, ['col2', 'col3'])
        self.assertListEqual(evaluation_model.metadata.categorical_columns, ['col1', 'col4'])
        self.assertListEqual(evaluation_model.metadata.numerical_columns, ['col2', 'col3'])







if __name__ == '__main__':
    unittest.main()
