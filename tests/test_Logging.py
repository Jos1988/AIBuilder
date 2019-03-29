import unittest
from pathlib import Path
from unittest import mock

from AIBuilder.AIFactory.Logging import LogRecord, MetaLogger, RecordCollection, CSVConverter, CSVReader


class TestLogRecord(unittest.TestCase):

    def test_is_same_group(self):
        r1 = LogRecord(attributes={'a': 1, 'b': 2, 'c': 'data', 'group': 5}, metrics={'m_a': 1, 'm_b': 2},
                       discrimination_value='group')
        r2 = LogRecord(attributes={'a': 1, 'b': 2, 'c': 'data', 'group': 6}, metrics={'m_a': 5, 'm_b': 3},
                       discrimination_value='group')
        r3 = LogRecord(attributes={'a': 7, 'b': 8, 'c': 'data', 'group': 7}, metrics={'m_a': 1, 'm_b': 2},
                       discrimination_value='group')

        self.assertTrue(r1.is_same_group(r2))
        self.assertTrue(r2.is_same_group(r1))

        self.assertFalse(r1.is_same_group(r3))
        self.assertFalse(r3.is_same_group(r1))


class TestMetaLogger(unittest.TestCase):
    model1_metrics = {'base_line': 0.5, 'accuracy': 0.8}
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
        'optimizer': {'gradient_clipping': None, 'kwargs': None, 'optimizer_learning_rate': 0.0,
                      'optimizer_type': 'null_optimizer'}, 'naming': {},
        'estimator': {'estimator_type': 'boosted_trees_classifier',
                      'kwargs': {'n_batches_per_layer': 5, 'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1,
                                 'l1_regularization': 0.0, 'l2_regularization': 0.0, 'tree_complexity': 0.1,
                                 'min_node_weight': 0.0, 'config': {'tf_random_seed': 2024}}}}

    model2_metrics = {'base_line': 0.5, 'accuracy': 0.9}
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
        'optimizer': {'gradient_clipping': None, 'kwargs': None, 'optimizer_learning_rate': 0.0,
                      'optimizer_type': 'null_optimizer'}, 'naming': {},
        'estimator': {'estimator_type': 'boosted_trees_classifier',
                      'kwargs': {'n_batches_per_layer': 5, 'n_trees': 250, 'max_depth': 6, 'learning_rate': 0.1,
                                 'l1_regularization': 0.0, 'l2_regularization': 0.0, 'tree_complexity': 0.1,
                                 'min_node_weight': 0.0, 'config': {'tf_random_seed': 2025}}}}

    model3_metrics = {'base_line': 0.5, 'accuracy': 0.75}
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
        'optimizer': {'gradient_clipping': None, 'kwargs': None, 'optimizer_learning_rate': 0.0,
                      'optimizer_type': 'null_optimizer'}, 'naming': {},
        'estimator': {'estimator_type': 'boosted_trees_classifier',
                      'kwargs': {'n_batches_per_layer': 5, 'n_trees': 251, 'max_depth': 7, 'learning_rate': 0.2,
                                 'l1_regularization': 0.0, 'l2_regularization': 0.0, 'tree_complexity': 0.1,
                                 'min_node_weight': 0.0, 'config': {'tf_random_seed': 2024}}}}

    def setUp(self):
        self.meta_logger = MetaLogger(log_attributes=['n_trees', 'max_depth', 'learning_rate'],
                                      log_metrics=['base_line', 'accuracy'],
                                      log_file_path=Path(''),
                                      discrimination_value='tf_random_seed')

    def test_create_record(self):
        record = self.meta_logger.create_record_from_dict(self.model1_description, self.model1_metrics)
        self.assertIsInstance(record, LogRecord)
        self.assertEqual({'n_trees': '250', 'max_depth': '6', 'learning_rate': '0.1', 'tf_random_seed': '2024'},
                         record.attributes)
        self.assertEqual({'base_line': '0.5', 'accuracy': '0.8'}, record.metrics)

    def test_create_multiple_records(self):
        model1 = mock.Mock('test_Logging.AI')
        model1.description = self.model1_description
        model1.results = self.model1_metrics
        model2 = mock.Mock('test_Logging.AI')
        model2.description = self.model2_description
        model2.results = self.model2_metrics
        model3 = mock.Mock('test_Logging.AI')
        model3.description = self.model3_description
        model3.results = self.model3_metrics

        self.meta_logger.log_ml_model(model1)
        self.meta_logger.log_ml_model(model2)
        self.meta_logger.log_ml_model(model3)

        groups = self.meta_logger.record_collection.record_groups

        self.assertEqual(2, len(groups))
        self.assertEqual(2, len(groups[0]))

        self.assertEqual({'n_trees': '250', 'max_depth': '6', 'learning_rate': '0.1', 'tf_random_seed': '2024'},
                         groups[0][0].attributes)
        self.assertEqual({'base_line': '0.5', 'accuracy': '0.8'}, groups[0][0].metrics)
        self.assertEqual({'n_trees': '250', 'max_depth': '6', 'learning_rate': '0.1', 'tf_random_seed': '2025'},
                         groups[0][1].attributes)
        self.assertEqual({'base_line': '0.5', 'accuracy': '0.9'}, groups[0][1].metrics)

        self.assertEqual(1, len(groups[1]))
        self.assertEqual({'n_trees': '251', 'max_depth': '7', 'learning_rate': '0.2', 'tf_random_seed': '2024'},
                         groups[1][0].attributes)
        self.assertEqual({'base_line': '0.5', 'accuracy': '0.75'}, groups[1][0].metrics)


def getTestRecords():
    record1a = LogRecord({'a': '1', 'b': '1', 'seed': '10'}, {'m_a': '1', 'm_b': '2'}, 'seed')
    record1b = LogRecord({'a': '1', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
    record1c = LogRecord({'a': '1', 'b': '1', 'seed': '12'}, {'m_a': '3', 'm_b': '2'}, 'seed')

    record2a = LogRecord({'a': '1', 'b': '2', 'seed': '10'}, {'m_a': '1', 'm_b': '2'}, 'seed')
    record2b = LogRecord({'a': '1', 'b': '2', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
    record2c = LogRecord({'a': '1', 'b': '2', 'seed': '12'}, {'m_a': '3', 'm_b': '2'}, 'seed')

    record3a = LogRecord({'a': '2', 'b': '1', 'seed': '10'}, {'m_a': '1', 'm_b': '2'}, 'seed')
    record3b = LogRecord({'a': '2', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
    record3c = LogRecord({'a': '2', 'b': '1', 'seed': '12'}, {'m_a': '3', 'm_b': '2'}, 'seed')

    record4 = LogRecord({'a': '2', 'b': '2', 'seed': '10'}, {'m_a': '1', 'm_b': '1'}, 'seed')

    return [record3a, record3b, record3c, record2a, record2b, record2c, record1a, record1b, record1c, record4]


class TestCSVConverter(unittest.TestCase):

    def test_generate_summary(self):
        record1 = LogRecord({'a': '2', 'b': '1', 'seed': '10'}, {'m_a': '1', 'm_b': '2.5'}, 'seed')
        record2 = LogRecord({'a': '2', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
        record3 = LogRecord({'a': '2', 'b': '1', 'seed': '12'}, {'m_a': '3', 'm_b': '1.5'}, 'seed')

        summary = CSVConverter.generate_summary([record1, record2, record3])
        self.assertEqual('2.0', summary['m_a'])
        self.assertEqual('2.0', summary['m_b'])


class TestRecordCollection(unittest.TestCase):

    def setUp(self):
        self.collection = RecordCollection(['a', 'b', 'seed'], ['m_a', 'm_b'], 'seed')
        self.records = getTestRecords()

    def test_add(self):
        for record in self.records:
            self.collection.add(record)

        groups = self.collection.record_groups

        self.assertEqual(3, len(groups[0]))
        self.assertEqual(3, len(groups[1]))
        self.assertEqual(3, len(groups[2]))
        self.assertEqual(1, len(groups[3]))

        self.assertEqual({'a': '1', 'b': '1', 'seed': '10'}, groups[2][0].attributes)
        self.assertEqual({'a': '1', 'b': '1', 'seed': '11'}, groups[2][1].attributes)
        self.assertEqual({'a': '1', 'b': '1', 'seed': '12'}, groups[2][2].attributes)

        self.assertEqual({'a': '1', 'b': '2', 'seed': '10'}, groups[1][0].attributes)
        self.assertEqual({'a': '1', 'b': '2', 'seed': '11'}, groups[1][1].attributes)
        self.assertEqual({'a': '1', 'b': '2', 'seed': '12'}, groups[1][2].attributes)

        self.assertEqual({'a': '2', 'b': '1', 'seed': '10'}, groups[0][0].attributes)
        self.assertEqual({'a': '2', 'b': '1', 'seed': '11'}, groups[0][1].attributes)
        self.assertEqual({'a': '2', 'b': '1', 'seed': '12'}, groups[0][2].attributes)

        self.assertEqual({'a': '2', 'b': '2', 'seed': '10'}, groups[3][0].attributes)

        self.assertEqual({'m_a': '1', 'm_b': '2'}, groups[2][0].metrics)
        self.assertEqual({'m_a': '2', 'm_b': '2'}, groups[2][1].metrics)
        self.assertEqual({'m_a': '3', 'm_b': '2'}, groups[2][2].metrics)

        self.assertEqual({'m_a': '1', 'm_b': '2'}, groups[1][0].metrics)
        self.assertEqual({'m_a': '2', 'm_b': '2'}, groups[1][1].metrics)
        self.assertEqual({'m_a': '3', 'm_b': '2'}, groups[1][2].metrics)

        self.assertEqual({'m_a': '1', 'm_b': '2'}, groups[0][0].metrics)
        self.assertEqual({'m_a': '2', 'm_b': '2'}, groups[0][1].metrics)
        self.assertEqual({'m_a': '3', 'm_b': '2'}, groups[0][2].metrics)

        self.assertEqual({'m_a': '1', 'm_b': '1'}, groups[3][0].metrics)

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


@unittest.skip
class TestCSVHandling(unittest.TestCase):
    def setUp(self):
        self.path = Path('data/test_log.csv')

    # Test creates a file.
    def test_write_meta_log(self):
        collection = RecordCollection(['a', 'b', 'seed'], ['m_a', 'm_b'], 'seed')

        for record in getTestRecords():
            collection.add(record)

        file = self.path.open(mode='w', newline='')
        self.csv_converter = CSVConverter(file=file, record_collection=collection)
        self.csv_converter.writeMetaLog()
        file.close()

    # Test requires file.
    def test_check_csv_compatible(self):
        reader = CSVReader(attribute_names=['a', 'b', 'seed'], metric_names=['m_a', 'm_b'], discriminator='seed')
        result = reader.check_compatible(self.path)

        self.assertEqual(True, result)

    # Test requires existing file.
    def test_metalogger_update_csv(self):
        metalogger = MetaLogger(log_attributes=['a', 'b', 'seed'], log_metrics=['m_a', 'm_b'],
                                discrimination_value='seed', log_file_path=Path('data/test_log.csv'))

        existing_records = RecordCollection(['a', 'b', 'seed'], ['m_a', 'm_b'], 'seed')
        existing_record1 = LogRecord({'a': '1', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
        existing_record2 = LogRecord({'a': '1', 'b': '2', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
        existing_record3 = LogRecord({'a': '2', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')

        existing_records.add(existing_record2)
        existing_records.add(existing_record3)
        existing_records.add(existing_record1)

        metalogger.record_collection = existing_records
        metalogger.save_to_csv()

    # Test requires file.
    def test_load_csv_to_records(self):
        file = self.path.open(mode='r', newline='')
        reader = CSVReader(attribute_names=['a', 'b', 'seed'], metric_names=['m_a', 'm_b'], discriminator='seed')
        result = reader.load_csv(file)
        expected = getTestRecords()

        self.assertEqual(len(result), len(expected))
        i = len(result) - 1
        records = result.get_records()

        while i >= 0:
            resulting_record = records[i]
            expected_record = expected[i]
            self.assertEqual(resulting_record.attributes, expected_record.attributes)
            self.assertEqual(resulting_record.metrics, expected_record.metrics)
            i = i - 1

        file.close()
        self.path.unlink()
