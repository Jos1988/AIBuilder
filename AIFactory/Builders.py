from abc import ABC, abstractmethod
from copy import deepcopy

from AIBuilder.AIFactory.EstimatorStrategies import EstimatorStrategyFactory, EstimatorStrategy
from AIBuilder.AIFactory.FeatureColumnStrategies import FeatureColumnStrategyFactory, FeatureColumnStrategy
from AIBuilder.AIFactory.OptimizerStrategies import OptimizerStrategyFactory, OptimizerStrategy
from AIBuilder.AIFactory.Specifications import Specification, ConfigDescriber, PrefixedDictSpecification
from AIBuilder.Data import DataModel, DataLoader, DataSetSplitter
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

    builder_registry = []

    @property
    @abstractmethod
    def dependent_on(self) -> list:
        pass

    @property
    @abstractmethod
    def builder_type(self) -> str:
        pass

    def __init__(self):
        self.builder_registry.append(self)

    @abstractmethod
    def validate(self):
        pass

    @classmethod
    def get_all_registered(cls):
        return cls.builder_registry

    def describe(self):
        description = {}
        specs = self.get_specs()
        for spec in specs:
            description[spec.name] = spec.describe()

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

    def __init__(self, data_source: str, target_column: str, data_columns: list, weight_column: str = None):
        super().__init__()
        self.data_columns = DataTypeSpecification('columns', data_columns, list)
        self.data_source = DataTypeSpecification('data_source', data_source, str)
        self.target_column = DataTypeSpecification('target_column', target_column, str)
        self.weight_column = NullSpecification('weight_column')

        if None is not weight_column:
            self.weight_column = DataTypeSpecification('weight_column', weight_column, str)

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
        data.set_weight_column(self.weight_column())
        ai.set_training_data(data)

    def load_data(self) -> DataModel:
        loader = DataLoader()

        self.load_file(loader)

        columns = self.data_columns().copy()
        columns.append(self.target_column())
        columns.append(self.weight_column())
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
        super().__init__()
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
        return [
            self.DATA_MODEL,  # Need data to split
            self.SCRUBBER,  # Scrub data as one data set because m_hot column scrubber adds new columns based on
        ]  # data in the current set.

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
    estimator = None
    estimator_type = None

    def __init__(self, estimator_type: str, config_kwargs: dict = None, kwargs: dict = None):
        """ build Estimator

        :param estimator_type: valid estimator strategy
        :param config_kwargs: pass the kwargs for a tf.estimator.RunConfig here, otherwise it will not be printed
        correctly in description.
        :param kwargs: estimator kwargs, do not pass objects if config is printed. //todo: find way to accurately print objects.
        """
        super().__init__()
        self.set_estimator(estimator_type)
        if config_kwargs is not None:
            self.config_kwargs = DataTypeSpecification('config_kwargs', config_kwargs, dict)
            kwargs = self.set_config(config_kwargs, kwargs)

        self.kwargs = NullSpecification('kwargs')
        if kwargs is not None:
            self.kwargs = DataTypeSpecification('kwargs', kwargs, dict)

    @staticmethod
    def set_config(config_kwargs, kwargs):
        config = ConfigDescriber(**config_kwargs)
        config.description = config_kwargs
        if kwargs is None:
            kwargs = {}

        kwargs.update({'config': config})

        return kwargs

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.NAMING_SCHEME, self.FEATURE_COLUMN]

    @property
    def builder_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.estimator_type = TypeSpecification('estimator_type', estimator_type, EstimatorStrategy.ALL_STRATEGIES)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        strategy: EstimatorStrategy = EstimatorStrategyFactory.get_strategy(neural_net,
                                                                            self.estimator_type(),
                                                                            self.kwargs())

        assert strategy is not None, 'Strategy for building Estimator of type {} not found.' \
            .format(self.estimator_type())

        self.estimator = strategy.build()
        neural_net.set_estimator(self.estimator)

        return

    @staticmethod
    def render_model_dir(ai) -> str:
        return ai.get_log_dir() + '/' + ai.get_project_name() + '/tensor_board/' + ai.name()


class InputFunctionBuilder(Builder):
    BASE_FN = 'base_fn'
    PANDAS_FN = 'pandas_fn'

    VALID_FN_NAMES = [BASE_FN, PANDAS_FN]

    def __init__(self, train_fn: str, train_kwargs: dict, evaluation_fn: str, evaluation_kwargs: dict):
        super().__init__()
        self.train_fn_name = TypeSpecification('test_dir function', train_fn, self.VALID_FN_NAMES)

        # todo: do not print training kwargs, when saving description, some of them are objects,
        #  for example x and y are dataframes.
        self.train_kwargs = train_kwargs
        self.train_kwargs_descr = PrefixedDictSpecification('train_kwargs', 'train', train_kwargs)

        self.evaluation_fn_name = TypeSpecification('evaluation function', evaluation_fn, self.VALID_FN_NAMES)
        # todo: idem.
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs_descr = PrefixedDictSpecification('evaluation_kwargs', 'eval', evaluation_kwargs)

        self.fn_holder = InputFunctionHolder

    @property
    def dependent_on(self) -> list:
        return [self.DATA_SPLITTER,  # Need both validation and training data separated for respective input fns.
                self.SCRUBBER,  # Pass clean data to input fns.
                self.FEATURE_COLUMN  # Feature column builder als add feature cols to data_models,
                #  which are required to for passing feature data to fn.
                ]

    @property
    def builder_type(self) -> str:
        return self.INPUT_FUNCTION

    def assign_fn(self, data_model: DataModel, fn_name: str, kwargs: dict):
        if hasattr(self.fn_holder, fn_name):
            return self.load_from_holder(data_model, fn_name, kwargs)

        if fn_name == self.PANDAS_FN:
            kwargs['x'] = data_model.get_input_fn_x_data()
            kwargs['y'] = data_model.get_target_column()
            kwargs['target_column'] = data_model.target_column_name

            fn = getattr(tf.estimator.inputs, 'pandas_input_fn')

            return fn(**kwargs)

    def load_from_holder(self, data_model: DataModel, fn_name: str, kwargs: dict):
        assert hasattr(self.fn_holder, fn_name), 'Function {} not known in function holder.'.format(fn_name)
        assert callable(getattr(self.fn_holder, fn_name)), 'Function {} is not callable.'.format(fn_name)
        fn = getattr(self.fn_holder, fn_name)
        kwargs['data_model'] = data_model

        return lambda: fn(**kwargs)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        train_function = self.assign_fn(neural_net.training_data, self.train_fn_name(), self.train_kwargs)
        evaluation_function = self.assign_fn(neural_net.evaluation_data, self.evaluation_fn_name(),
                                             self.evaluation_kwargs)

        neural_net.set_training_fn(train_function)
        neural_net.set_evaluation_fn(evaluation_function)


class NamingSchemeBuilder(Builder):

    def __init__(self):
        super().__init__()
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

    def __init__(self, optimizer_type: str, learning_rate: float, gradient_clipping: Optional[float] = None,
                 kwargs: dict = None):
        super().__init__()
        self.optimizer_type = TypeSpecification('optimizer_type', optimizer_type, OptimizerStrategy.ALL_STRATEGIES)
        self.learning_rate = DataTypeSpecification('optimizer_learning_rate', learning_rate, float)

        self.gradient_clipping = NullSpecification('gradient_clipping')
        if gradient_clipping is not None:
            self.gradient_clipping = DataTypeSpecification('gradient_clipping', gradient_clipping, float)

        self.kwargs = NullSpecification('kwargs')
        if kwargs is not None:
            self.kwargs = DataTypeSpecification('kwargs', kwargs, dict)

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.OPTIMIZER

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        strategy: OptimizerStrategy = OptimizerStrategyFactory.get_strategy(neural_net,
                                                                            self.optimizer_type(),
                                                                            self.learning_rate(),
                                                                            self.gradient_clipping(),
                                                                            self.kwargs())

        assert strategy is not None, 'Strategy for building Optimizer of type {} not found.' \
            .format(self.optimizer_type())

        optimizer = strategy.build()
        neural_net.set_optimizer(optimizer)

        return


class ScrubAdapter(Builder):

    def __init__(self, scrubbers: list = None):
        super().__init__()
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
            self.and_scrubber.validate_metadata(deepcopy(training_data.metadata))
            self.and_scrubber.scrub(training_data)

        if validation_data is not None:
            self.and_scrubber.validate_metadata(deepcopy(validation_data.metadata))
            self.and_scrubber.scrub(validation_data)


class MetadataBuilder(Builder):

    def __init__(self, overrules: dict = {}):
        super().__init__()
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
                if overrule not in metadata.column_collections:
                    raise RuntimeError('Metadata overwrite {}, not recognized for column {}.'.format(overrule, column))

                metadata.add_column_to_type(column_type=overrule, column_name=column)
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
                metadata.define_unknown_columns([column])
                raise RuntimeError('Column {} with value {} cannot be categorized.'.format(column, data_type))

        return metadata


class FeatureColumnBuilder(Builder):

    def __init__(self, feature_columns: dict, feature_config: dict = None):
        super().__init__()
        self.feature_config = feature_config
        self.feature_columns = FeatureColumnsSpecification('feature_columns', [], FeatureColumnStrategy.ALL_COLUMNS)

        for name, type in feature_columns.items():
            if type is FeatureColumnStrategy.BUCKETIZED_COLUMN:
                assert name in feature_config, 'Missing configuration for bucketized column: {}.'.format(name)
                assert 'buckets' in self.feature_config[name], \
                    'Missing buckets configuration for bucketized column: {}.'.format(name)

            self.add_feature_column(name=name, column_type=type)

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL, self.SCRUBBER]
    # requires scrubbed data in order to generate categories for categorical columns.

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

    def build_feature_columns(self, data_model):
        self.validate_target_not_in_features(data_model)

        feature_columns = self.render_tf_feature_columns(data_model=data_model)
        data_model.set_tf_feature_columns(feature_columns)

        self.add_feature_col_names_to_data_model(data_model)

    def add_feature_col_names_to_data_model(self, data_model):
        for column_data in self.feature_columns():

            if column_data['type'] is FeatureColumnStrategy.MULTIPLE_HOT_COLUMNS:
                self.load_bin_column_names(column_data['name'], data_model)

                continue

            data_model.add_feature_column(column_data['name'])

    @staticmethod
    def load_bin_column_names(column_name: str, data_model):
        binary_column_prefix = column_name + '_'
        # prefix convention implemented in 'MultipleCatListToMultipleHotScrubber'.
        for column in data_model.get_dataframe().columns:
            if binary_column_prefix in column:
                data_model.add_feature_column(column)

    def validate_target_not_in_features(self, data: DataModel):
        target_column = data.target_column_name
        assert target_column not in self.feature_columns(), 'Feature column \'{}\' is already set as target column' \
            .format(target_column)

    def render_tf_feature_columns(self, data_model: DataModel) -> list:
        tf_feature_columns = []
        for feature_column_info in self.feature_columns():
            column_strategy = FeatureColumnStrategyFactory.get_strategy(feature_column_info['name'],
                                                                        feature_column_info['type'],
                                                                        data_model,
                                                                        self.feature_config)
            feature_columns = column_strategy.build()
            tf_feature_columns = feature_columns + tf_feature_columns

        return tf_feature_columns
