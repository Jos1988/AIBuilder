from abc import ABC, abstractmethod
from copy import deepcopy

from AIBuilder.AIFactory.EstimatorStrategies import EstimatorStrategyFactory, EstimatorStrategy
from AIBuilder.AIFactory.FeatureColumnStrategies import FeatureColumnStrategyFactory, FeatureColumnStrategy
from AIBuilder.AIFactory.OptimizerStrategies import OptimizerStrategyFactory, OptimizerStrategy
from AIBuilder.AIFactory.Specifications import ConfigDescriber, PrefixedDictSpecification, Describer
from AIBuilder.AIFactory.smartCache.SmartCache import smart_cache
from AIBuilder.Data import DataModel, DataLoader, DataSetSplitter
import tensorflow as tf
import pandas as pd
import AIBuilder.InputFunctionHolder as InputFunctionHolder
import numpy as np
from typing import Optional, List
from AIBuilder.AIFactory.Specifications import DataTypeSpecification, TypeSpecification, \
    NullSpecification, RangeSpecification, Descriptor, FeatureColumnsSpecification
from AIBuilder.AI import AbstractAI
import AIBuilder.DataScrubbing as scrubber


class Builder(ABC, Describer):
    # add you new builder types here.
    ESTIMATOR = 'estimator'
    OPTIMIZER = 'optimizer'
    DATA_MODEL = 'data_model'
    FEATURE_COLUMN = 'feature_column'
    META_DATA = 'meta_data'
    SCRUBBER = 'scrubber'
    INPUT_FUNCTION = 'input_function'
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

    @abstractmethod
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        """ ML_model is changed by the builder, we also return te result for the sake of caching. """
        pass


class DataBuilder(Builder):

    def __init__(self, target_column: str, data_columns: list, data_source: str = None, eval_data_source: str = None,
                 prediction_data_source: str = None, weight_column: str = None, limit: int = 0):
        super().__init__()
        self.data_columns = DataTypeSpecification('columns', data_columns, list)
        self.target_column = DataTypeSpecification('target_column', target_column, str)
        self.weight_column = NullSpecification('weight_column')

        self.data_source = NullSpecification('data_source')
        self.eval_data_source = NullSpecification('eval_data_source')
        self.prediction_data_source = NullSpecification('prediction_data_source')
        self.limit = NullSpecification('limit')

        if None is not data_source:
            self.data_source = DataTypeSpecification('data_source', data_source, str)

        if None is not eval_data_source:
            self.eval_data_source = DataTypeSpecification('eval_data_source', eval_data_source, str)

        if None is not prediction_data_source:
            self.prediction_data_source = DataTypeSpecification('prediction_data_source', prediction_data_source, str)

        if None is not weight_column:
            self.weight_column = DataTypeSpecification('weight_column', weight_column, str)

        if None is not limit:
            self.limit = DataTypeSpecification('limit', limit, int)

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

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        if self.data_source() is not None:
            data = self.load_data(self.data_source())
            ml_model.set_training_data(data)

        if self.eval_data_source() is not None:
            validation_data = self.load_data(self.eval_data_source())
            ml_model.set_evaluation_data(validation_data)

        if self.prediction_data_source() is not None:
            prediction_data = self.load_data(self.prediction_data_source())
            ml_model.set_prediction_data(prediction_data)

        return ml_model

    def load_data(self, data_source: str) -> DataModel:
        loader = self.load_file(data_source)

        columns = self.data_columns().copy()
        columns.append(self.target_column())
        columns.append(self.weight_column())
        loader.filter_columns(set(columns))

        data = loader.get_dataset()
        data.set_target_column(self.target_column())
        data.set_weight_column(self.weight_column())

        return data

    def load_prediction_data(self, data_source: str) -> DataModel:
        loader = self.load_file(data_source)

        columns = self.data_columns().copy()
        columns.append(self.weight_column())
        loader.filter_columns(set(columns))

        data = loader.get_dataset()
        data.set_weight_column(self.weight_column())

        return data

    def load_file(self, data_source: str) -> DataLoader:
        loader = DataLoader(self.limit())

        if 'csv' in data_source:
            loader.load_csv(data_source)
            return loader

        raise RuntimeError('Failed to load data from {}.'.format(self.data_source()))


class DataSplitterBuilder(Builder):
    TRAINING_DATA = 'training'
    EVALUATION_DATA = 'evaluation'

    def __init__(self, data_source: str = None, random_seed: int = None):
        super().__init__()
        self.data_source = NullSpecification('data_source')
        if data_source is not None:
            self.data_source = TypeSpecification(name='data_source',
                                                 value=data_source,
                                                 valid_types=[self.TRAINING_DATA, self.EVALUATION_DATA])
        self.randomize = DataTypeSpecification('splitter_randomize', False, bool)
        self.seed = NullSpecification('splitter_seed')
        if random_seed is not None:
            self.randomize = DataTypeSpecification('splitter_randomize', True, bool)
            self.seed = DataTypeSpecification('splitter_seed', random_seed, int)

    @staticmethod
    def randomize_data(data: DataModel, seed: int):
        df = data.get_dataframe()
        df = df.sample(frac=1, random_state=seed)
        data.set_dataframe(df)

    def select_data(self, neural_net: AbstractAI) -> DataModel:
        if self.data_source() == self.TRAINING_DATA:
            data = neural_net.get_training_data()
        elif self.data_source() == self.EVALUATION_DATA:
            data = neural_net.get_evaluation_data()
        else:
            raise RuntimeError('Unknown data_source ({}) found in data splitter builder.'.format(self.data_source()))
        return data


class NullDataSplitterBuilder(DataSplitterBuilder):

    def __init__(self, random_seed: int = None):
        super().__init__(None, random_seed)

    def validate(self):
        pass

    @property
    def dependent_on(self) -> list:
        return [
            self.DATA_MODEL,  # Need data to split
            self.SCRUBBER,  # Scrub data as one data set because m_hot column scrubber adds new columns based on
        ]  # data in the current set.

    @property
    def builder_type(self) -> str:
        return self.DATA_SPLITTER

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        if not self.randomize():
            return ml_model

        train_data = ml_model.get_training_data()
        self.randomize_data(train_data, self.seed())
        ml_model.set_training_data(training_data=train_data)

        eval_data = ml_model.get_evaluation_data()
        if eval_data is None:
            return ml_model

        self.randomize_data(eval_data, self.seed())
        ml_model.set_evaluation_data(eval_data)

        return ml_model


class RandomDataSplitter(DataSplitterBuilder):

    def __init__(self, evaluation_data_perc: int, data_source: str, random_seed: Optional[int] = None):
        """ Splits data already set on the ml model, using either the training data or evaluation data as source.
        The respective data is then split and loaded back into the training and evaluation data of the model.

        :param evaluation_data_perc: Percentage of data that will be cut of from data in data source and set to
                                     evaluation data of the ml model.
        :param data_source:          Data used to split.
        :param random_seed:          If set, seed will be used to shuffle the data before splitting, run multiple
                                     models with different seeds to achieve k-fold evaluation.
        """
        super().__init__(data_source, random_seed)
        self.randomize = DataTypeSpecification('splitter_randomize', False, bool)
        self.seed = NullSpecification('splitter_seed')
        if random_seed is not None:
            self.randomize = DataTypeSpecification('splitter_randomize', True, bool)
            self.seed = DataTypeSpecification('splitter_seed', random_seed, int)

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

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        data = self.select_data(ml_model)
        if self.randomize():
            self.randomize_data(data, self.seed())

        splitter = DataSetSplitter(data_model=data)
        split_data = splitter.split_by_ratio([self.training_data_percentage(), self.evaluation_data_perc()])

        ml_model.set_training_data(split_data[0])
        ml_model.set_evaluation_data(split_data[1])

        return ml_model

    def select_data(self, neural_net: AbstractAI) -> DataModel:
        if self.data_source() == self.TRAINING_DATA:
            data = neural_net.get_training_data()
        elif self.data_source() == self.EVALUATION_DATA:
            data = neural_net.get_evaluation_data()
        else:
            raise RuntimeError('Unknown data_source ({}) found in data splitter builder.'.format(self.data_source()))
        return data


class CategoricalDataSplitter(DataSplitterBuilder):
    TRAINING_DATA = 'training'
    EVALUATION_DATA = 'evaluation'

    def __init__(self, data_source: str, column_name: str, training_categories: List[str] = None,
                 eval_categories: List[str] = None, verbosity: int = 0):
        """ Splits data already set on the ml model, using either the training data or evaluation data as source.
        The respective data is split using categories from a given column

        :param data_source:          Data used to split.
        """
        super().__init__(data_source, None)
        self.verbosity = verbosity
        self.data_source = TypeSpecification(name='data_source',
                                             value=data_source,
                                             valid_types=[self.TRAINING_DATA, self.EVALUATION_DATA])

        self.column_name = DataTypeSpecification(name='column_name', value=column_name, data_type=str)
        self.training_categories = NullSpecification('training_categories')
        if training_categories is not None:
            self.training_categories = DataTypeSpecification('training_categories', training_categories, list)

        self.eval_categories = NullSpecification('eval_categories')
        if eval_categories is not None:
            self.eval_categories = DataTypeSpecification('eval_categories', eval_categories, list)

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

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        data_model = self.select_data(ml_model)
        df = data_model.get_dataframe()

        training_cats = self.training_categories()
        eval_cats = self.eval_categories()
        all_cats = df[self.column_name()].unique()

        eval_cats, training_cats = self.set_categories(all_cats, eval_cats, training_cats)

        training_data = df[df[self.column_name()].isin(training_cats)]
        ml_model.set_training_data(self.load_data(data_model, training_data))

        evaluation_data = df[df[self.column_name()].isin(eval_cats)]
        ml_model.set_evaluation_data(self.load_data(data_model, evaluation_data))

        if self.verbosity > 0:
            print(f'{len(training_data)} items in training data.')
            print(f'{len(evaluation_data)} items in evaluation data.')

        return ml_model

    @staticmethod
    def set_categories(all_cats, eval_cats, training_cats):
        assert training_cats is not None or eval_cats is not None, 'Categories for either training or evaluation ' \
                                                                   'data must be set.'
        if training_cats is None:
            training_cats = [cat for cat in all_cats if cat not in eval_cats]
        if eval_cats is None:
            eval_cats = [cat for cat in all_cats if cat not in training_cats]
        assert set(training_cats + eval_cats) == set(all_cats), f'Some categories are unaccounted for found: ' \
                                                                f'{training_cats + eval_cats}, need {all_cats}'
        return eval_cats, training_cats

    @staticmethod
    def load_data(data_model: DataModel, new_data: pd.DataFrame) -> DataModel:
        new_data_model = deepcopy(data_model)
        new_data_model.set_dataframe(new_data)

        return new_data_model


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
            self.kwargs = PrefixedDictSpecification('kwargs', 'est', kwargs)

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
        return [self.OPTIMIZER, self.FEATURE_COLUMN]

    @property
    def builder_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.estimator_type = TypeSpecification('estimator_type', estimator_type, EstimatorStrategy.ALL_STRATEGIES)

    def validate(self):
        self.validate_specifications()

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        strategy: EstimatorStrategy = EstimatorStrategyFactory.get_strategy(ml_model,
                                                                            self.estimator_type(),
                                                                            self.kwargs())

        assert strategy is not None, 'Strategy for building Estimator of type {} not found.' \
            .format(self.estimator_type())

        self.estimator = strategy.build()
        ml_model.set_estimator(self.estimator)

        return ml_model


class InputFunctionBuilder(Builder):
    BASE_FN = 'base_fn'
    PANDAS_FN = 'pandas_fn'

    VALID_FN_NAMES = [BASE_FN, PANDAS_FN]

    def __init__(self, train_fn: str = None, train_kwargs: dict = None, evaluation_fn: str = None,
                 evaluation_kwargs: dict = None,
                 prediction_fn: str = None, prediction_kwargs: dict = None):
        super().__init__()
        self.fn_holder = InputFunctionHolder

        self.build_train = False
        if train_fn is not None:
            self.build_train = True
            self.train_fn_name = TypeSpecification('test_dir function', train_fn, self.VALID_FN_NAMES)
            # todo: do not print training kwargs, when saving description, some of them are objects,
            #  for example x and y are dataframes.
            self.train_kwargs = train_kwargs
            self.train_kwargs_descr = PrefixedDictSpecification('train_kwargs', 'train', train_kwargs)

        self.build_eval = False
        if evaluation_fn is not None:
            self.build_eval = True
            self.evaluation_fn_name = TypeSpecification('evaluation function', evaluation_fn, self.VALID_FN_NAMES)
            # todo: idem.
            self.evaluation_kwargs = evaluation_kwargs
            self.evaluation_kwargs_descr = PrefixedDictSpecification('evaluation_kwargs', 'eval', evaluation_kwargs)

        self.build_predict = False
        if prediction_fn is not None:
            self.build_predict = True
            self.prediction_fn_name = TypeSpecification('prediction function', prediction_fn, self.VALID_FN_NAMES)
            # todo: idem.
            self.prediction_kwargs = prediction_kwargs
            self.prediction_kwargs_descr = PrefixedDictSpecification('prediction_kwargs', 'pred', prediction_kwargs)

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

    def assign_prediction_fn(self, data_model: DataModel, fn_name: str, kwargs: dict):
        if hasattr(self.fn_holder, fn_name):
            return self.load_from_holder(data_model, fn_name, kwargs)

        if fn_name == self.PANDAS_FN:
            kwargs['x'] = data_model.get_input_fn_x_data()

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

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        if self.build_train:
            train_function = self.assign_fn(ml_model.training_data, self.train_fn_name(), self.train_kwargs)
            ml_model.set_training_fn(train_function)

        if self.build_eval:
            evaluation_function = self.assign_fn(ml_model.evaluation_data, self.evaluation_fn_name(),
                                                 self.evaluation_kwargs)
            ml_model.set_evaluation_fn(evaluation_function)

        if self.build_predict:
            prediction_function = self.assign_prediction_fn(ml_model.prediction_data, self.prediction_fn_name(),
                                                            self.prediction_kwargs)
            ml_model.set_prediction_fn(prediction_function)

        return ml_model


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

    @smart_cache
    def build(self, ml_model: AbstractAI):
        strategy: OptimizerStrategy = OptimizerStrategyFactory.get_strategy(ml_model,
                                                                            self.optimizer_type(),
                                                                            self.learning_rate(),
                                                                            self.gradient_clipping(),
                                                                            self.kwargs())

        assert strategy is not None, 'Strategy for building Optimizer of type {} not found.' \
            .format(self.optimizer_type())

        optimizer = strategy.build()
        ml_model.set_optimizer(optimizer)

        return ml_model


class ScrubAdapter(Builder):
    TRAINING_DATA = 'training_data'
    EVALUATION_DATA = 'evaluation_data'
    PREDICTION_DATA = 'prediction_data'

    VALID_DATA_TARGET = [TRAINING_DATA, EVALUATION_DATA, PREDICTION_DATA]

    def __init__(self, scrubbers: list = None):
        super().__init__()
        self.and_scrubber_training = scrubber.AndScrubber()
        self.and_scrubber_validation = scrubber.AndScrubber()
        self.and_scrubber_prediction = scrubber.AndScrubber()
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

    def add_scrubber(self, scrubber: scrubber.Scrubber, scrubber_target: str = None):
        """ Add a scrubber to both and_scrubbers, if a target is specified the scrubber will only be added to the
        respective and_scrubber
        """

        assert scrubber_target is None or scrubber_target in self.VALID_DATA_TARGET, \
            'Invalid data target: {}'.format(scrubber_target)

        if scrubber_target is None or scrubber_target is self.TRAINING_DATA:
            self.and_scrubber_training.add_scrubber(scrubber)

        if scrubber_target is None or scrubber_target is self.EVALUATION_DATA:
            self.and_scrubber_validation.add_scrubber(scrubber)

        if scrubber_target is None or scrubber_target is self.PREDICTION_DATA:
            self.and_scrubber_prediction.add_scrubber(scrubber)

        description_postfix = ''
        if scrubber_target is not None:
            description_postfix = ' (' + scrubber_target + ')'

        self.descriptor.add_description(scrubber.__class__.__name__ + description_postfix)

    def describe(self):
        return {'scrubber_names': super(ScrubAdapter, self).describe(),
                'scrubbers': self.and_scrubber_training.describe()}

    def validate(self):
        pass

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        training_data = ml_model.training_data
        validation_data = ml_model.evaluation_data
        prediction_data = ml_model.get_prediction_data()

        if training_data is not None:
            self.and_scrubber_training.validate_metadata(deepcopy(training_data.metadata))
            self.and_scrubber_training.scrub(training_data)

        if validation_data is not None:
            self.and_scrubber_validation.validate_metadata(deepcopy(validation_data.metadata))
            self.and_scrubber_validation.scrub(validation_data)

        if prediction_data is not None:
            self.and_scrubber_prediction.validate_metadata(deepcopy(prediction_data.metadata))
            self.and_scrubber_prediction.scrub(prediction_data)

        return ml_model


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

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        training_data_model: Optional[DataModel] = ml_model.get_training_data()
        evaluation_data_model: Optional[DataModel] = ml_model.get_evaluation_data()
        prediction_data_model: Optional[DataModel] = ml_model.get_prediction_data()

        if None is not training_data_model:
            ml_model.set_training_data(self.build_meta_data(training_data_model))

        if None is not evaluation_data_model:
            ml_model.set_evaluation_data(self.build_meta_data(evaluation_data_model))

        if None is not prediction_data_model:
            ml_model.set_prediction_data(self.build_meta_data(prediction_data_model))

        return ml_model

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
    MULTIPLE_COLUMNS_FEATURE = [
        FeatureColumnStrategy.INDICATOR_COLUMN_VOC_LIST,
        FeatureColumnStrategy.VECTOR_COLUMNS
    ]

    def __init__(self, feature_columns: dict, feature_config: dict = None):
        super().__init__()
        self.feature_config = DataTypeSpecification('feature_config', feature_config, dict)
        self.feature_columns = FeatureColumnsSpecification('feature_columns', [], FeatureColumnStrategy.ALL_COLUMNS)

        for name, type in feature_columns.items():
            if type is FeatureColumnStrategy.BUCKETIZED_COLUMN:
                assert name in feature_config, 'Missing configuration for bucketized column: {}.'.format(name)
                assert 'buckets' in self.feature_config()[name], \
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

    @smart_cache
    def build(self, ml_model: AbstractAI) -> AbstractAI:
        training_data: Optional[DataModel] = ml_model.get_training_data()
        evaluation_data: Optional[DataModel] = ml_model.get_evaluation_data()
        prediction_data: Optional[DataModel] = ml_model.get_prediction_data()

        if training_data is not None:
            self.build_feature_columns(training_data)
            ml_model.set_training_data(training_data)

        if evaluation_data is not None:
            self.build_feature_columns(evaluation_data)
            ml_model.set_evaluation_data(evaluation_data)

        if prediction_data is not None:
            self.build_feature_columns(prediction_data)
            ml_model.set_evaluation_data(prediction_data)

        return ml_model

    def build_feature_columns(self, data_model):
        self.validate_target_not_in_features(data_model)

        self.render_tf_feature_columns(data_model=data_model)
        self.add_feature_col_names_to_data_model(data_model)

    def add_feature_col_names_to_data_model(self, data_model):
        for column_data in self.feature_columns():

            if column_data['type'] in self.MULTIPLE_COLUMNS_FEATURE:
                self.load_columns_by_prefix(column_data['name'], data_model)
                continue

            if column_data['type'] is FeatureColumnStrategy.CROSSED_COLUMN:
                continue

            data_model.add_feature_column(column_data['name'])

    @staticmethod
    def load_columns_by_prefix(column_name: str, data_model):
        # prefix convention implemented in 'MultipleCatListToMultipleHotScrubber'.
        column_prefix = column_name + '_'
        columns = [column for column in data_model.get_dataframe().columns if column_prefix in column]
        for column in columns:
            data_model.add_feature_column(column)

    def validate_target_not_in_features(self, data: DataModel):
        target_column = data.target_column_name
        assert target_column not in self.feature_columns(), 'Feature column \'{}\' is already set as target column' \
            .format(target_column)

    def render_tf_feature_columns(self, data_model: DataModel):
        data_model.set_tf_feature_columns([])
        for feature_column_info in self.feature_columns():
            column_strategy = FeatureColumnStrategyFactory.get_strategy(feature_column_info['name'],
                                                                        feature_column_info['type'],
                                                                        data_model,
                                                                        self.feature_config())
            feature_columns = column_strategy.build()
            for tf_feature_column in feature_columns:
                data_model.add_tf_feature_columns(tf_feature_column)
