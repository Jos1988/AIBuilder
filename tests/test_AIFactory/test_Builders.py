from unittest import mock
from AIBuilder.AI import AI, AbstractAI
from AIBuilder.AIFactory.EstimatorStrategies import EstimatorStrategy
from AIBuilder.AIFactory.FeatureColumnStrategies import FeatureColumnStrategy
from AIBuilder.AIFactory.OptimizerStrategies import OptimizerStrategy
from AIBuilder.Data import DataModel
from AIBuilder.AIFactory.Specifications import TypeSpecification
import unittest
import tensorflow as tf
import pandas as pd
from AIBuilder.AIFactory.Builders import Builder
import AIBuilder.DataScrubbing as scrubber
from AIBuilder.AIFactory.Builders import DataBuilder, EstimatorBuilder, InputFunctionBuilder, NamingSchemeBuilder, \
    OptimizerBuilder, ScrubAdapter, MetadataBuilder, DataSplitterBuilder, FeatureColumnBuilder


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
        self.specification_one.describe = mock.Mock()
        self.specification_one.describe.return_value = 'value1'
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
                                   data_columns=['feature_1', 'feature_2', 'feature_3'])

        arti = AI(project_name='name', log_dir='path/to/dir')
        data_builder.validate()
        data_builder.build(ai=arti)

        column_names = ['feature_1', 'feature_2', 'feature_3', 'target_1']
        self.validate_data_frame(arti.training_data, column_names)

    def test_constructor_build(self):
        data_builder = DataBuilder(data_source='C:/python/practice2/AIBuilder/tests/data/test_data.csv',
                                   target_column='target_1',
                                   data_columns=['feature_1', 'feature_2', 'feature_3'])

        arti = AI(project_name='name', log_dir='path/to/dir')
        data_builder.validate()
        data_builder.build(ai=arti)

        column_names = ['feature_1', 'feature_2', 'feature_3', 'target_1']
        self.validate_data_frame(arti.training_data, column_names)

    def validate_data_frame(self, data_frame: DataModel, columns: list):
        self.assertEqual(data_frame.target_column_name, 'target_1')
        self.assertCountEqual(data_frame.get_dataframe().columns.tolist(), columns, )


class TestEstimatorBuilder(unittest.TestCase):

    def test_validate(self):
        estimator_builder = EstimatorBuilder(EstimatorStrategy.LINEAR_REGRESSOR)
        estimator_builder.validate()

    def test_invalid_estimator_type(self):
        invalid_estimator_builder = EstimatorBuilder(EstimatorStrategy.LINEAR_REGRESSOR)
        invalid_estimator_builder.estimator_type = TypeSpecification(name=EstimatorBuilder.ESTIMATOR,
                                                                     value='invalid',
                                                                     valid_types=EstimatorBuilder.valid_estimator_types)

        valid_estimator_builder = EstimatorBuilder(EstimatorStrategy.LINEAR_REGRESSOR)

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

        estimator_builder = EstimatorBuilder(EstimatorStrategy.LINEAR_REGRESSOR, config_kwargs={'tf_random_seed': 123})

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
        self.assertEqual(123, estimator_builder.estimator.config.tf_random_seed)


class TestInputFnBuilder(unittest.TestCase):

    def setUp(self):
        data_array = {'feat_A': [1, 2, 3], 'feat_B': [8, 6, 4], 'target': [9, 8, 7]}
        df2 = df = pd.DataFrame(data_array)

        train_model = DataModel(df)
        train_model.set_target_column('target')
        train_model.set_feature_columns(['feat_A', 'feat_B'])

        eval_model = DataModel(df2)
        eval_model.set_target_column('target')
        eval_model.set_feature_columns(['feat_A', 'feat_B'])

        self.arti = AI('test', 'test/test')
        self.arti.training_data = train_model
        self.arti.evaluation_data = eval_model

    def set_base_fn(self):
        self.input_fn_builder = InputFunctionBuilder(train_fn=InputFunctionBuilder.BASE_FN,
                                                     train_kwargs={'batch_size': 100, 'epoch': 1},
                                                     evaluation_fn=InputFunctionBuilder.BASE_FN,
                                                     evaluation_kwargs={'batch_size': 100, 'epoch': 1})

    def test_validate_base(self):
        self.set_base_fn()
        self.input_fn_builder.validate()

    def test_build_base(self):
        self.set_base_fn()
        self.input_fn_builder.build(self.arti)

        self.assertEqual(self.arti.training_fn.__name__, '<lambda>')

    def set_pandas_fn(self):
        self.input_fn_builder = InputFunctionBuilder(train_fn=InputFunctionBuilder.PANDAS_FN,
                                                     train_kwargs={'num_epochs': 1, 'shuffle': True},
                                                     evaluation_fn=InputFunctionBuilder.PANDAS_FN,
                                                     evaluation_kwargs={'num_epochs': 1, 'shuffle': False})

    def test_validate_pandas(self):
        self.set_pandas_fn()
        self.input_fn_builder.validate()

    def test_build_pandas(self):
        self.set_pandas_fn()
        self.input_fn_builder.build(self.arti)
        self.assertEqual(self.arti.training_fn.__name__, 'input_fn')


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
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
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
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=0.1)

        optimizer_builder_with_clipping.build(arti)
        arti.set_optimizer.assert_called_once()
        mock_optimizer.assert_called_with(learning_rate=1.0)
        mock_clipper.assert_called()

    def test_build_with_no_clipping(self, ):
        arti = mock.Mock('OptimizerBuilder.AbstractAI')
        arti.set_optimizer = mock.Mock()

        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_no_clipping.build(arti)
        arti.set_optimizer.assert_called_once()


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

        and_scrubber.validate_metadata.assert_called(),
        and_scrubber.scrub.assert_any_call(training_data),
        and_scrubber.scrub.assert_any_call(evaluation_data)


class TestMetadataBuilder(unittest.TestCase):

    def test_build(self):
        builder = MetadataBuilder({'col5': 'unknown'})
        arti = mock.Mock('AIBuilder.AbstractAI')

        # mock evaluation model
        data = {'col1': ['cat1', 'cat2'], 'col2': [3, 4], 'col3': [0.1, 0.2], 'col4': [True, False],
                'col5': ['some', 'thing']}
        dataframe = pd.DataFrame(data=data)
        evaluation_model = DataModel(dataframe)

        arti.get_evaluation_data = mock.Mock()
        arti.get_evaluation_data.return_value = evaluation_model
        arti.set_evaluation_data = mock.Mock()

        # mock training model
        data = {'col1': ['cat1', 'cat2'], 'col2': [3, 4], 'col3': [0.1, 0.2], 'col4': [True, False],
                'col5': ['some', 'thing']}
        dataframe = pd.DataFrame(data=data)
        training_model = DataModel(dataframe)

        arti.get_training_data = mock.Mock()
        arti.get_training_data.return_value = training_model
        arti.set_training_data = mock.Mock()

        # build
        builder.build(arti)

        # assert
        self.assertListEqual(training_model.metadata.categorical_columns, ['col1', 'col4'])
        self.assertListEqual(training_model.metadata.numerical_columns, ['col2', 'col3'])
        self.assertListEqual(training_model.metadata.uncategorized_columns, ['col5'])
        self.assertListEqual(evaluation_model.metadata.categorical_columns, ['col1', 'col4'])
        self.assertListEqual(evaluation_model.metadata.numerical_columns, ['col2', 'col3'])
        self.assertListEqual(evaluation_model.metadata.uncategorized_columns, ['col5'])
        arti.set_evaluation_data.assert_called_once()
        arti.set_training_data.assert_called_once()


class TestDataSplitterBuilder(unittest.TestCase):

    def test_build(self):
        builder = DataSplitterBuilder(evaluation_data_perc=20, data_source=DataSplitterBuilder.TRAINING_DATA)
        arti = mock.Mock('AIBuilder.AbstractAI')

        # mock training model
        data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]}
        dataframe = pd.DataFrame(data=data)
        training_model = DataModel(dataframe)

        arti.get_training_data = mock.Mock()
        arti.get_training_data.return_value = training_model
        arti.set_training_data = mock.Mock()

        # mock evaluation model
        data = {'col1': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                'col2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        dataframe = pd.DataFrame(data=data)
        evaluation_model = DataModel(dataframe)

        arti.get_evaluation_data = mock.Mock()
        arti.get_evaluation_data.return_value = evaluation_model
        arti.set_evaluation_data = mock.Mock()

        builder.build(neural_net=arti)

        arti.set_evaluation_data.assert_called_once()
        arti.set_training_data.assert_called_once()

        split_evaluation_data = arti.set_evaluation_data.call_args[0][0].get_dataframe()
        split_training_data = arti.set_training_data.call_args[0][0].get_dataframe()

        self.assertEqual(2, len(split_evaluation_data))
        self.assertEqual(8, len(split_training_data))


class TestFeatureColumnBuilder(unittest.TestCase):

    def test_build(self):
        builder = FeatureColumnBuilder(
            feature_columns={
                'col1': FeatureColumnStrategy.CATEGORICAL_COLUMN_VOC_LIST,
                'col3': FeatureColumnStrategy.NUMERICAL_COLUMN,
                'col4': FeatureColumnStrategy.INDICATOR_COLUMN_VOC_LIST,
                'col5': FeatureColumnStrategy.BUCKETIZED_COLUMN,
            },
            buckets={'col5': 2}
        )

        arti = mock.Mock('AIBuilder.AbstractAI')

        # mock training model
        data = {'col3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                'col1': ['cat_one', 'cat_one', 'cat_one', 'cat_one', 'cat_one', 'cat_two', 'cat_one', 'cat_two',
                         'cat_one', 'cat_two'],
                'col4': [
                    ['cat_four', 'cat_three', 'cat_four'],
                    ['cat_two', 'cat_three', 'cat_four'],
                    ['cat_two', 'cat_three', 'cat_four'],
                    ['cat_two', 'cat_four'],
                    ['cat_one', 'cat_three', 'cat_four'],
                    ['cat_one', 'cat_three', 'cat_four'],
                    ['cat_one', 'cat_three'],
                    ['cat_one', 'cat_three', 'cat_four'],
                    ['cat_one', 'cat_two', 'cat_three'],
                    ['cat_two', 'cat_three'],
                ],
                'col5': [1, 2, 3, 4, 1, 2, 3, 4]}

        dataframe = pd.DataFrame(data=data)
        training_model = DataModel(dataframe)
        training_model.get_target_column = mock.Mock()
        training_model.get_target_column.return_value = 'col2'

        arti.get_training_data = mock.Mock()
        arti.get_training_data.return_value = training_model
        arti.set_training_data = mock.Mock()

        arti.get_evaluation_data = mock.Mock()
        arti.get_evaluation_data.return_value = None

        builder.build(arti)

        feature_columns = training_model.get_tf_feature_columns()
        col1_cat_column = feature_columns[2]
        col3_num_column = feature_columns[1]
        col4_indicator_column = feature_columns[0]

        arti.set_training_data.assert_called_once()

        self.assertCountEqual(col1_cat_column.vocabulary_list, {'cat_one', 'cat_two'})
        self.assertEqual(col1_cat_column.name, 'col1')

        self.assertEqual(col3_num_column.name, 'col3')
        self.assertEqual(col3_num_column.dtype, tf.float32)

        self.assertEqual(col4_indicator_column.name, 'col4_indicator')

    def test_missing_bucket_config(self):
        with self.assertRaises(AssertionError):
            FeatureColumnBuilder(
                feature_columns={
                    'col1': FeatureColumnStrategy.BUCKETIZED_COLUMN,
                },
                feature_config={
                    'wrong column': {}
                }
            )

        with self.assertRaises(AssertionError):
            FeatureColumnBuilder(
                feature_columns={
                    'col1': FeatureColumnStrategy.BUCKETIZED_COLUMN,
                },
                feature_config={
                    'col1': {'wrong_config': 1}
                }
            )


if __name__ == '__main__':
    unittest.main()
