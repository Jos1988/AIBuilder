from abc import ABC, abstractmethod
from AIBuilder.AIFactory.Specifications import Specification
from AIBuilder.Data import MetaData, DataModel, DataLoader, DataSetSplitter
import tensorflow as tf
import AIBuilder.InputFunctionHolder as InputFunctionHolder
import os
import numpy as np
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
    FEATURE_COLUMN = 'feature_column'
    META_DATA = 'meta_data'
    SCRUBBER = 'scrubber'
    INPUT_FUNCTION = 'input_function'
    NAMING_SCHEME = 'naming'
    DATA_SPLITTER = 'data_splitter'

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

    def __init__(self, data_source: str,
                 target_column: str,
                 data_columns: list):

        self.data_columns = DataTypeSpecification('columns', data_columns, list)
        self.data_source = DataTypeSpecification('data_source', data_source, str)
        self.target_column = DataTypeSpecification('target_column', target_column, str)
        self.test_data = None
        self.validation_data = None

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.DATA_MODEL

    def validate(self):
        self.validate_specifications()

    def build(self, ai: AbstractAI):
        data = self.load_data()
        data.set_target_column(self.target_column())
        ai.set_training_data(data)

    def load_data(self) -> DataModel:
        loader = DataLoader()

        self.load_file(loader)

        columns = self.data_columns()
        columns.append(self.target_column())
        loader.filter_columns(set(columns))

        return loader.get_dataset()

    def load_file(self, loader: DataLoader):
        if 'csv' in self.data_source():
            loader.load_csv(self.data_source())
            return

        raise RuntimeError('Failed to load data from {}.'.format(self.data_source()))


class DataSplitterBuilder(Builder):
    TRAINING_DATA = 'training'
    EVALUATION_DATA = 'evaluation'

    def __init__(self, evaluation_data_perc: int, data_source: str):
        self.data_source = TypeSpecification(name='data_source',
                                             value=data_source,
                                             valid_types=[self.TRAINING_DATA, self.EVALUATION_DATA])
        self.evaluation_data_perc = RangeSpecification(name='evaluation_data_perc',
                                                       value=evaluation_data_perc,
                                                       min_value=0,
                                                       max_value=100)
        self.training_data_percentage = RangeSpecification(name='training_data_perc',
                                                           value=(100 - evaluation_data_perc),
                                                           min_value=0,
                                                           max_value=100)

    @property
    def dependent_on(self) -> list:
        return [self.META_DATA, self.DATA_MODEL, self.SCRUBBER, self.FEATURE_COLUMN]

    @property
    def builder_type(self) -> str:
        return self.DATA_SPLITTER

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        data = self.select_data(neural_net)

        splitter = DataSetSplitter(data_model=data)
        split_data = splitter.split_by_ratio([self.training_data_percentage(), self.evaluation_data_perc()])

        neural_net.set_training_data(split_data[0])
        neural_net.set_evaluation_data(split_data[1])

    def select_data(self, neural_net):
        if self.data_source() == self.TRAINING_DATA:
            data = neural_net.get_training_data()
        elif self.data_source() == self.EVALUATION_DATA:
            data = neural_net.get_training_data()
        else:
            raise RuntimeError('Unknown data_source ({}) found in data splitter builder.'.format(self.data_source()))
        return data


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
        return [self.DATA_MODEL, self.META_DATA]

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

        if training_data is not None:
            self.and_scrubber.validate_metadata(training_data.metadata)
            self.and_scrubber.scrub(training_data)

        if validation_data is not None:
            self.and_scrubber.validate_metadata(validation_data.metadata)
            self.and_scrubber.scrub(validation_data)


class MetadataBuilder(Builder):

    def __init__(self, overrules: dict = {}):
        self.overrules = overrules

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.META_DATA

    def validate(self):
        pass

    def build(self, neural_net: AbstractAI):
        training_data_model: Optional[DataModel] = neural_net.get_training_data()
        evaluation_data_model: Optional[DataModel] = neural_net.get_evaluation_data()

        if None is not training_data_model:
            neural_net.set_training_data(self.build_meta_data(training_data_model))

        if None is not evaluation_data_model:
            neural_net.set_evaluation_data(self.build_meta_data(evaluation_data_model))

    def build_meta_data(self, data_model):
        df = data_model.get_dataframe()
        types = df.dtypes.to_dict()
        data_model.metadata = self.fill_metadata(data_model.metadata, types)

        return data_model

    def fill_metadata(self, metadata, types):
        for column, data_type in types.items():

            if column in self.overrules:
                overrule = self.overrules[column]
                if overrule == 'numerical':
                    metadata.define_numerical_columns([column])
                elif overrule == 'categorical':
                    metadata.define_categorical_columns([column])
                elif overrule == 'unknown':
                    metadata.define_uncategorized_columns([column])
                else:
                    raise RuntimeError('Metadata overwrite {}, not recognized for column {}.'.format(overrule, column))

                continue

            if data_type == object:
                metadata.define_categorical_columns([column])
            elif data_type in [np.dtype('float32'), np.dtype('float64')]:
                metadata.define_numerical_columns([column])
            elif data_type in [np.dtype('int32'), np.dtype('int64')]:
                metadata.define_numerical_columns([column])
            elif data_type == bool:
                metadata.define_categorical_columns([column])
            else:
                metadata.define_uncategorized_columns([column])
                raise RuntimeError('Column {} with value {} cannot be categorized.'.format(column, data_type))

        return metadata


class FeatureColumnBuilder(Builder):
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'
    NUMERICAL_COLUMN = 'numeric_column'
    valid_column_types = [CATEGORICAL_COLUMN_VOC_LIST, NUMERICAL_COLUMN]

    def __init__(self, feature_columns: dict):

        self.feature_columns = FeatureColumnsSpecification('feature_columns', [], self.valid_column_types)

        for name, type in feature_columns.items():
            self.add_feature_column(name=name, column_type=type)

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL, self.SCRUBBER]

    @property
    def builder_type(self) -> str:
        return self.FEATURE_COLUMN

    def add_feature_column(self, name: str, column_type: str):
        self.feature_columns.add_feature_column(name=name, column_type=column_type)

    def validate(self):
        self.validate_specifications()

    def build(self, ai: AbstractAI):
        training_data: Optional[DataModel] = ai.get_training_data()
        evaluation_data: Optional[DataModel] = ai.get_evaluation_data()

        if training_data is not None:
            self.build_feature_columns(training_data)
            ai.set_training_data(training_data)

        if evaluation_data is not None:
            self.build_feature_columns(evaluation_data)
            ai.set_evaluation_data(evaluation_data)

    def build_feature_columns(self, training_data):
        self.validate_target_not_in_features(training_data)
        feature_columns = self.render_tf_feature_columns(data=training_data)
        training_data.set_tf_feature_columns(feature_columns)

    def validate_target_not_in_features(self, data: DataModel):
        target_column = data.target_column_name
        assert target_column not in self.feature_columns(), 'Feature column \'{}\' is already set as target column' \
            .format(target_column)

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

        assert None not in categories, 'None in not a valid category, please check your scrubbing! None found in {}'\
            .format(feature_column_data['name'])

        return tf.feature_column.categorical_column_with_vocabulary_list(
            feature_column_data['name'],
            vocabulary_list=categories
        )