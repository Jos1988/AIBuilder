import unittest
from unittest import mock

from AIBuilder.AIFactory.Logging import LogRecord, MetaLogger, RecordCollection, CSVConverter


class TestLogRecord(unittest.TestCase):

    def test_is_same_group(self):
        r1 = LogRecord(values={'a': 1, 'b': 2, 'c': 'data', 'group': 5}, discrimination_value='group')
        r2 = LogRecord(values={'a': 1, 'b': 2, 'c': 'data', 'group': 6}, discrimination_value='group')
        r3 = LogRecord(values={'a': 7, 'b': 8, 'c': 'data', 'group': 7}, discrimination_value='group')

        self.assertTrue(r1.is_same_group(r2))
        self.assertTrue(r2.is_same_group(r1))

        self.assertFalse(r1.is_same_group(r3))
        self.assertFalse(r3.is_same_group(r1))


class TestMetaLogger(unittest.TestCase):
    model1_description = {
        'data_model': {'columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived'],
                       'data_source': '../data/titanic/train_800.csv', 'target_column': 'Survived'}, 'meta_data': {},
        'scrubber': {
            'scrubbers': ['MissingDataScrubber', 'ConvertToNumericScrubber', 'OutlierScrubber', 'BinaryResampler']},
        'data_splitter': {'data_source': 'training', 'evaluation_data_perc': 20, 'training_data_perc': 80},
        'feature_column': {'feature_columns': [{'name': 'Pclass', 'type': 'indicator_column'},
                                               {'name': 'Sex', 'type': 'indicator_column'},
                                               {'name': 'Parch', 'type': 'bucketized_column'},
                                               {'name': 'Fare', 'type': 'bucketized_column'}]},
        'input_function': {'evaluation function': 'pandas_fn',
                           'evaluation_kwargs': {'num_epochs': 1, 'batch_size': 1, 'shuffle': False,
                                                 'target_column': 'Survived'}, 'test_dir function': 'pandas_fn',
                           'train_kwargs': {'num_epochs': 10, 'batch_size': 10, 'shuffle': True,
                                            'target_column': 'Survived'}},
        # 'optimizer': {'gradient_clipping': None, 'kwargs': None, 'learning_rate': 0.0,
        #               'optimizer_type': 'null_optimizer'}, 'naming': {},
        'estimator': {'estimator_type': 'boosted_trees_classifier',
                      'kwargs': {'n_batches_per_layer': 5, 'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1,
                                 'l1_regularization': 0.0, 'l2_regularization': 0.0, 'tree_complexity': 0.1,
                                 'min_node_weight': 0.0, 'config': {'tf_random_seed': 2024}}}}

    model2_description = {
        'data_model': {'columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived'],
                       'data_source': '../data/titanic/train_800.csv', 'target_column': 'Survived'}, 'meta_data': {},
        'scrubber': {
            'scrubbers': ['MissingDataScrubber', 'ConvertToNumericScrubber', 'OutlierScrubber', 'BinaryResampler']},
        'data_splitter': {'data_source': 'training', 'evaluation_data_perc': 20, 'training_data_perc': 80},
        'feature_column': {'feature_columns': [{'name': 'Pclass', 'type': 'indicator_column'},
                                               {'name': 'Sex', 'type': 'indicator_column'},
                                               {'name': 'Parch', 'type': 'bucketized_column'},
                                               {'name': 'Fare', 'type': 'bucketized_column'}]},
        'input_function': {'evaluation function': 'pandas_fn',
                           'evaluation_kwargs': {'num_epochs': 1, 'batch_size': 1, 'shuffle': False,
                                                 'target_column': 'Survived'}, 'test_dir function': 'pandas_fn',
                           'train_kwargs': {'num_epochs': 10, 'batch_size': 10, 'shuffle': True,
                                            'target_column': 'Survived'}},
        # 'optimizer': {'gradient_clipping': None, 'kwargs': None, 'learning_rate': 0.0,
        #               'optimizer_type': 'null_optimizer'}, 'naming': {},
        'estimator': {'estimator_type': 'boosted_trees_classifier',
                      'kwargs': {'n_batches_per_layer': 5, 'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1,
                                 'l1_regularization': 0.0, 'l2_regularization': 0.0, 'tree_complexity': 0.1,
                                 'min_node_weight': 0.0, 'config': {'tf_random_seed': 2025}}}}

    model3_description = {
        'data_model': {'columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived'],
                       'data_source': '../data/titanic/train_800.csv', 'target_column': 'Survived'}, 'meta_data': {},
        'scrubber': {
            'scrubbers': ['MissingDataScrubber', 'ConvertToNumericScrubber', 'OutlierScrubber', 'BinaryResampler']},
        'data_splitter': {'data_source': 'training', 'evaluation_data_perc': 20, 'training_data_perc': 80},
        'feature_column': {'feature_columns': [{'name': 'Pclass', 'type': 'indicator_column'},
                                               {'name': 'Sex', 'type': 'indicator_column'},
                                               {'name': 'Parch', 'type': 'bucketized_column'},
                                               {'name': 'Fare', 'type': 'bucketized_column'}]},
        'input_function': {'evaluation function': 'pandas_fn',
                           'evaluation_kwargs': {'num_epochs': 1, 'batch_size': 1, 'shuffle': False,
                                                 'target_column': 'Survived'}, 'test_dir function': 'pandas_fn',
                           'train_kwargs': {'num_epochs': 10, 'batch_size': 10, 'shuffle': True,
                                            'target_column': 'Survived'}},
        # 'optimizer': {'gradient_clipping': None, 'kwargs': None, 'learning_rate': 0.0,
        #               'optimizer_type': 'null_optimizer'}, 'naming': {},
        'estimator': {'estimator_type': 'boosted_trees_classifier',
                      'kwargs': {'n_batches_per_layer': 5, 'n_trees': 251, 'max_depth': 7, 'learning_rate': 0.2,
                                 'l1_regularization': 0.0, 'l2_regularization': 0.0, 'tree_complexity': 0.1,
                                 'min_node_weight': 0.0, 'config': {'tf_random_seed': 2024}}}}

    def setUp(self):
        self.meta_logger = MetaLogger(log_values=['n_trees', 'max_depth', 'learning_rate'], log_file_path='',
                                      discrimination_value='tf_random_seed')

    def test_create_record(self):
        record = self.meta_logger.create_record_from_dict(self.model1_description)
        self.assertIsInstance(record, LogRecord)
        self.assertEqual({'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1, 'tf_random_seed': 2024}, record.values)

    def test_create_multiple_records(self):
        model1 = mock.Mock('test_Logging.AI')
        model1.description = self.model1_description
        model2 = mock.Mock('test_Logging.AI')
        model2.description = self.model2_description
        model3 = mock.Mock('test_Logging.AI')
        model3.description = self.model3_description

        self.meta_logger.log_ml_model(model1)
        self.meta_logger.log_ml_model(model2)
        self.meta_logger.log_ml_model(model3)

        groups = self.meta_logger.record_collection.record_groups

        self.assertEqual(2, len(groups))
        self.assertEqual(2, len(groups[0]))

        self.assertEqual({'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1, 'tf_random_seed': 2024},
                         groups[0][0].values)
        self.assertEqual({'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1, 'tf_random_seed': 2025},
                         groups[0][1].values)

        self.assertEqual(1, len(groups[1]))
        self.assertEqual({'n_trees': 251, 'max_depth': 7, 'learning_rate': 0.2, 'tf_random_seed': 2024},
                         groups[1][0].values)


def getTestRecords():
    record1a = LogRecord({'a': 1, 'b': 1, 'seed': 10}, 'seed')
    record1b = LogRecord({'a': 1, 'b': 1, 'seed': 11}, 'seed')
    record1c = LogRecord({'a': 1, 'b': 1, 'seed': 12}, 'seed')

    record2a = LogRecord({'a': 1, 'b': 2, 'seed': 10}, 'seed')
    record2b = LogRecord({'a': 1, 'b': 2, 'seed': 11}, 'seed')
    record2c = LogRecord({'a': 1, 'b': 2, 'seed': 12}, 'seed')

    record3a = LogRecord({'a': 2, 'b': 1, 'seed': 10}, 'seed')
    record3b = LogRecord({'a': 2, 'b': 1, 'seed': 11}, 'seed')
    record3c = LogRecord({'a': 2, 'b': 1, 'seed': 12}, 'seed')

    record4 = LogRecord({'a': 2, 'b': 2, 'seed': 10}, 'seed')

    return [record3a, record3b, record3c, record2a, record2b, record2c, record1a, record1b, record1c, record4]


class TestRecordCollection(unittest.TestCase):

    def setUp(self):
        self.collection = RecordCollection(['a', 'b', 'seed'])

        self.records = getTestRecords()

    def test_add(self):
        for record in self.records:
            self.collection.add(record)

        groups = self.collection.record_groups

        self.assertEqual(3, len(groups[0]))
        self.assertEqual(3, len(groups[1]))
        self.assertEqual(3, len(groups[2]))
        self.assertEqual(1, len(groups[3]))

        self.assertEqual({'a': 1, 'b': 1, 'seed': 10}, groups[2][0].values)
        self.assertEqual({'a': 1, 'b': 1, 'seed': 11}, groups[2][1].values)
        self.assertEqual({'a': 1, 'b': 1, 'seed': 12}, groups[2][2].values)

        self.assertEqual({'a': 1, 'b': 2, 'seed': 10}, groups[1][0].values)
        self.assertEqual({'a': 1, 'b': 2, 'seed': 11}, groups[1][1].values)
        self.assertEqual({'a': 1, 'b': 2, 'seed': 12}, groups[1][2].values)

        self.assertEqual({'a': 2, 'b': 1, 'seed': 10}, groups[0][0].values)
        self.assertEqual({'a': 2, 'b': 1, 'seed': 11}, groups[0][1].values)
        self.assertEqual({'a': 2, 'b': 1, 'seed': 12}, groups[0][2].values)

        self.assertEqual({'a': 2, 'b': 2, 'seed': 10}, groups[3][0].values)

    def test_has(self):
        self.collection.add(self.records[0])
        self.collection.add(self.records[1])
        self.collection.add(self.records[2])
        self.collection.add(self.records[3])
        self.assertTrue(self.collection.has(self.records[0]))
        self.assertTrue(self.collection.has(self.records[3]))
        self.assertFalse(self.collection.has(self.records[5]))

    def test_remove(self):
        self.collection.add(self.records[0])
        self.collection.add(self.records[1])
        self.collection.add(self.records[2])
        self.collection.add(self.records[3])

        self.collection.remove(self.records[1])

        self.assertTrue(self.collection.has(self.records[0]))
        self.assertFalse(self.collection.has(self.records[1]))
        self.assertTrue(self.collection.has(self.records[3]))

        self.collection.remove(self.records[0])
        self.collection.remove(self.records[2])
        self.collection.remove(self.records[3])

        self.assertEqual([], self.collection.record_groups)


class TestCVSConverter(unittest.TestCase):

    @unittest.skip('Test disable as it create a file.')
    def test_write_meta_log(self):
        path = 'data/test_log.csv'
        collection = RecordCollection(['a', 'b', 'seed'])

        for record in getTestRecords():
            collection.add(record)

        file = open(path, mode='w', newline='')
        self.csv_converter = CSVConverter(file=file, record_collection=collection)
        self.csv_converter.writeMetaLog()
        file.close()
