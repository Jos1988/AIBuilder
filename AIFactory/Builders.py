from abc import ABC, abstractmethod
from AIBuilder.AIFactory.Specifications import Specification
from AIBuilder.Data import MetaData, DataModel, DataLoader, DataSetSplitter
import tensorflow as tf
import AIBuilder.InputFunctionHolder as InputFunctionHolder
import os
from typing import Optional
from AIBuilder.AIFactory.Specifications import DataTypeSpecification, TypeSpecification, \
    NullSpecification, RangeSpecification, Descriptor, FeatureColumnsSpecification
from AIBuilder.AI import AbstractAI
import AIBuilder.DataScrubbing as scrubber


class Builder(ABC):
    # add you new builder types here.
    ESTIMATOR = 'estimator'
    OPTIMIZER = 'optimizer'
    DATA_MODEL = 'data_model'
    SCRUBBER = 'scrubber'
    INPUT_FUNCTION = 'input_function'
    NAMING_SCHEME = 'naming'

    @property
    @abstractmethod
    def dependent_on(self) -> list:
        pass

    @property
    @abstractmethod
    def builder_type(self) -> str:
        pass

    @abstractmethod
    def validate(self):
        pass

    def describe(self):
        description = {}
        specs = self.get_specs()
        for spec in specs:
            description[spec.name] = spec.value

        return description

    def validate_specifications(self):
        specs = self.get_specs()

        for spec in specs:
            spec.validate()

    def get_specs(self):
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), Specification)]

    @abstractmethod
    def build(self, neural_net: AbstractAI):
        pass


class DataBuilder(Builder):
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'
    NUMERICAL_COLUMN = 'numeric_column'
    valid_column_types = [CATEGORICAL_COLUMN_VOC_LIST, NUMERICAL_COLUMN]

    def __init__(self, data_source: str,
                 target_column: str,
                 validation_data_percentage: int,
                 feature_columns: dict,
                 data_columns: list,
                 metadata: MetaData):

        self.data_columns = data_columns
        self.metadata = metadata
        self.validation_data_percentage = RangeSpecification('validation_data_perc', validation_data_percentage, 0, 100)
        self.training_data_percentage = RangeSpecification(name='training_data_perc',
                                                           value=(100 - self.validation_data_percentage.value),
                                                           min_value=0,
                                                           max_value=100)

        self.data_source = DataTypeSpecification('data_source', data_source, str)
        self.target_column = DataTypeSpecification('target_column', target_column, str)
        self.feature_columns = FeatureColumnsSpecification('feature_columns', [], self.valid_column_types)
        self.test_data = None
        self.validation_data = None

        # validate input.
        for name, type in feature_columns.items():
            self.add_feature_column(name=name, column_type=type)

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.DATA_MODEL

    def add_feature_column(self, name: str, column_type: str):
        self.feature_columns.add_feature_column(name=name, column_type=column_type)

    def validate(self):
        self.validate_specifications()
        assert self.target_column not in self.get_feature_column_names(), \
            'target column {}, also set as feature column!'.format(self.target_column())

    def build(self, ai: AbstractAI):
        data = self.load_data()
        data.set_target_column(self.target_column())
        data.metadata = self.metadata

        feature_columns = self.render_tf_feature_columns(data=data)
        data.set_tf_feature_columns(feature_columns)

        split_data = self.split_validation_and_test_data(data=data)
        ai.set_evaluation_data(split_data['validation_data'])
        ai.set_training_data(split_data['training_data'])

    def load_data(self) -> DataModel:
        loader = DataLoader()

        self.load_file(loader)

        columns = self.get_feature_column_names()
        columns.append(self.target_column())
        columns = columns + self.data_columns
        loader.filter_columns(columns)

        return loader.get_dataset()

    def load_file(self, loader: DataLoader):
        if 'csv' in self.data_source():
            loader.load_csv(self.data_source())
            return

        raise RuntimeError('Failed to load data from {}.'.format(self.data_source()))

    def get_feature_column_names(self) -> list:
        names = []
        for feature_column in self.feature_columns():
            names.append(feature_column['name'])

        return names

    def split_validation_and_test_data(self, data: DataModel):
        splitter = DataSetSplitter(data_model=data)
        result = splitter.split_by_ratio([self.training_data_percentage(), self.validation_data_percentage()])

        return {'training_data': result[0], 'validation_data': result[1]}

    # todo: possible separate builder
    def render_tf_feature_columns(self, data: DataModel) -> list:
        tf_feature_columns = []
        for feature_column in self.feature_columns():
            column = None
            if feature_column['type'] is self.CATEGORICAL_COLUMN_VOC_LIST:
                column = self.build_categorical_column_voc_list(feature_column, data)
            elif feature_column['type'] is self.NUMERICAL_COLUMN:
                column = self.build_numerical_column(feature_column['name'])

            if column is None:
                raise RuntimeError('feature column not set, ({})'.format(feature_column))

            tf_feature_columns.append(column)

        return tf_feature_columns

    @staticmethod
    def build_numerical_column(feature_column: dict) -> tf.feature_column.numeric_column:
        return tf.feature_column.numeric_column(feature_column)

    @staticmethod
    def build_categorical_column_voc_list(
            feature_column_data: dict,
            data: DataModel
    ) -> tf.feature_column.categorical_column_with_vocabulary_list:

        categories = data.get_all_column_categories(feature_column_data['name'])

        # todo: refactor so tf columns are manufactured in different builder after scrubbing.
        filtered_categories = [cat for cat in categories if type(cat) is str]

        return tf.feature_column.categorical_column_with_vocabulary_list(
            feature_column_data['name'],
            vocabulary_list=filtered_categories
        )


class EstimatorBuilder(Builder):
    LINEAR_REGRESSOR = 'linear_regressor'
    valid_estimator_types = [LINEAR_REGRESSOR]
    estimator: str
    estimator_type = None

    def __init__(self, estimator_type: str):
        self.set_estimator(estimator_type)

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.DATA_MODEL, self.NAMING_SCHEME]

    @property
    def builder_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.estimator_type = TypeSpecification('estimator_type', estimator_type, self.valid_estimator_types)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        if self.estimator_type() is self.LINEAR_REGRESSOR:
            estimator = tf.estimator.LinearRegressor(
                feature_columns=neural_net.training_data.get_tf_feature_columns(),
                optimizer=neural_net.optimizer,
                model_dir=self.render_model_dir(neural_net)
            )

            neural_net.set_estimator(estimator)
            return

        raise RuntimeError('Estimator Builder failed to set estimator.')

    @staticmethod
    def render_model_dir(ai) -> str:
        return ai.get_log_dir() + '/' + ai.get_project_name() + '/tensor_board/' + ai.get_name()


class InputFunctionBuilder(Builder):
    BASE_FN = 'base_fn'

    VALID_FN_NAMES = [BASE_FN]

    def __init__(self, train_fn: str, train_kwargs: dict, evaluation_fn: str, evaluation_kwargs: dict):
        self.train_fn_name = TypeSpecification('test_dir function', train_fn, self.VALID_FN_NAMES)
        self.train_kwargs = DataTypeSpecification('train_kwargs', train_kwargs, dict)

        self.evaluation_fn_name = TypeSpecification('evaluation function', evaluation_fn, self.VALID_FN_NAMES)
        self.evaluation_kwargs = DataTypeSpecification('evaluation_kwargs', evaluation_kwargs, dict)

        self.fn_holder = InputFunctionHolder

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.INPUT_FUNCTION

    def assign_fn(self, fn_name: str, kwargs: dict):
        assert hasattr(self.fn_holder, fn_name), 'Function {} not known in function holder.'.format(fn_name)
        assert callable(getattr(self.fn_holder, fn_name)), 'Function {} is not callable.'.format(fn_name)
        fn = getattr(self.fn_holder, fn_name)

        return lambda: fn(**kwargs)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        self.train_kwargs.value['data_model'] = neural_net.training_data
        self.evaluation_kwargs.value['data_model'] = neural_net.evaluation_data

        train_function = self.assign_fn(self.train_fn_name(), self.train_kwargs())
        evaluation_function = self.assign_fn(self.evaluation_fn_name(), self.evaluation_kwargs())

        neural_net.set_training_fn(train_function)
        neural_net.set_evaluation_fn(evaluation_function)

class NamingSchemeBuilder(Builder):

    def __init__(self):
        self.versions = []
        self.AI = None
        self.existing_names = []

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.NAMING_SCHEME

    def validate(self):
        pass

    def build(self, neural_net: AbstractAI):
        self.AI = neural_net
        self.existing_names = self.get_logged_names()

        if self.AI.get_name() is None or self.AI.get_name() is self.AI.get_project_name():
            self.generate_name()
            return

        if self.AI.get_name() in self.existing_names:
            self.AI.set_name(self.AI.get_name() + '_1')
            return

        if self.AI.get_name() is not None:
            return

        raise RuntimeError('Naming scheme failed to set name.')

    def generate_name(self):
        for name in self.existing_names:
            version = self.get_version(name=name)
            if version is not False:
                self.versions.append(version)

        last_version = 0
        if len(self.versions) > 0:
            last_version = max(self.versions)

        new_version = last_version + 1
        name = self.AI.get_project_name() + '_' + str(new_version)
        self.AI.set_name(name)

    def get_logged_names(self):
        tensor_board_path = self.AI.get_log_dir() + '/' + self.AI.get_project_name() + '/tensor_board'
        return next(os.walk(tensor_board_path))[1]

    def get_version(self, name: str):
        exploded = name.split('_')

        if exploded[0] == self.AI.get_project_name() and len(exploded) > 1 and exploded[1].isnumeric():
            return int(exploded[1])

        return False


class OptimizerBuilder(Builder):
    LEARNING_RATE = 'learning_rate'
    GRADIENT_CLIPPING = 'gradient_clipping'

    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

    valid_optimizer_types = [GRADIENT_DESCENT_OPTIMIZER]

    def __init__(self, optimizer_type: str, learning_rate: float, gradient_clipping: Optional[float] = None):
        self.optimizer_type = TypeSpecification('optimizer_type', optimizer_type, self.valid_optimizer_types)
        self.learning_rate = DataTypeSpecification('learning_rate', learning_rate, float)

        self.gradient_clipping = NullSpecification('gradient_clipping')
        if gradient_clipping is not None:
            self.gradient_clipping = DataTypeSpecification('gradient_clipping', gradient_clipping, float)

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.OPTIMIZER

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        my_optimizer = self._set_optimizer(optimizer_type=self.optimizer_type(), learning_rate=self.learning_rate())

        if self.gradient_clipping() is not None:
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, self.gradient_clipping())

        neural_net.set_optimizer(my_optimizer)

    def _set_optimizer(self, optimizer_type: str, learning_rate: float) -> tf.train.Optimizer:
        if optimizer_type is self.GRADIENT_DESCENT_OPTIMIZER:
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            return my_optimizer

        raise RuntimeError('Optimizer not set.')

class ScrubAdapter(Builder):

    def __init__(self, scrubbers: list = None):
        self.and_scrubber = scrubber.AndScrubber()
        self.descriptor = Descriptor('scrubbers', None)
        if scrubbers is not None:
            for new_scrubber in scrubbers:
                assert isinstance(new_scrubber, scrubber.Scrubber)
                self.add_scrubber(new_scrubber)

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.SCRUBBER

    def add_scrubber(self, scrubber: scrubber.Scrubber):
        self.and_scrubber.add_scrubber(scrubber)
        self.descriptor.add_description(scrubber.__class__.__name__)

    def validate(self):
        pass

    def build(self, neural_net: AbstractAI):
        training_data = neural_net.training_data
        validation_data = neural_net.evaluation_data

        self.and_scrubber.validate_metadata(training_data.metadata)
        self.and_scrubber.scrub(training_data)

        self.and_scrubber.validate_metadata(validation_data.metadata)
        self.and_scrubber.scrub(validation_data)
