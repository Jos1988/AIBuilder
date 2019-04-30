import unittest
import warnings
from pathlib import Path
from unittest import mock

from AIBuilder.AIFactory.Logging import LogRecord, RecordCollection, CSVFormatter, CSVReader, MetaLogger, \
    CSVMetaLogFormatter, CSVSummaryLogFormatter, SummaryLogger


class TestLogRecord(unittest.TestCase):

    def test_is_same_group(self):
        r1 = LogRecord(attributes={'a': 1, 'b': 2, 'c': 'data', 'group': 5}, metrics={'m_a': 1, 'm_b': 2},
                       discriminator_name='group')
        r2 = LogRecord(attributes={'a': 1, 'b': 2, 'c': 'data', 'group': 6}, metrics={'m_a': 5, 'm_b': 3},
                       discriminator_name='group')
        r3 = LogRecord(attributes={'a': 7, 'b': 8, 'c': 'data', 'group': 7}, metrics={'m_a': 1, 'm_b': 2},
                       discriminator_name='group')

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

        self.meta_logger.add_ml_model(model1)
        self.meta_logger.add_ml_model(model2)
        self.meta_logger.add_ml_model(model3)

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


class TestCSVMetaLogFormatter(unittest.TestCase):

    def test_generate_summary(self):
        record1 = LogRecord({'a': '2', 'b': '1', 'seed': '10'}, {'m_a': '1', 'm_b': '2.5'}, 'seed')
        record2 = LogRecord({'a': '2', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
        record3 = LogRecord({'a': '2', 'b': '1', 'seed': '12'}, {'m_a': '3', 'm_b': '1.5'}, 'seed')

        summary = CSVMetaLogFormatter.generate_group_summary([record1, record2, record3], write_attributes=False)
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

    def test_clear(self):
        for record in self.records:
            self.collection.add(record)

        self.assertEqual(4, len(self.collection.record_groups))
        self.collection.clear()
        self.assertEqual(0, len(self.collection.record_groups))


# @unittest.skip
class TestCSVHandling(unittest.TestCase):

    def load_test_collection(self):
        collection = RecordCollection(['a', 'b', 'seed'], ['m_a', 'm_b'], 'seed')
        for record in getTestRecords():
            collection.add(record)
        return collection

    def load_new_records(self):
        new_records = RecordCollection(['a', 'b', 'seed'], ['m_a', 'm_b'], 'seed')
        new_record1 = LogRecord({'a': '1', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
        new_record2 = LogRecord({'a': '1', 'b': '2', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')
        new_record3 = LogRecord({'a': '2', 'b': '1', 'seed': '11'}, {'m_a': '2', 'm_b': '2'}, 'seed')

        new_records.add(new_record2)
        new_records.add(new_record3)
        new_records.add(new_record1)

        return new_records

    def test_metalog_monolithic(self):
        self.metaLogPath = Path('data/test_meta_log.csv')
        self._write_meta_log()
        self._check_metalog_csv_compatible()
        self._meta_logger_update_csv()
        self._load_metalog_to_records()
        self.metaLogPath.unlink()

    # Test creates the metaLog file.
    def _write_meta_log(self):
        collection = self.load_test_collection()

        metaLog = self.metaLogPath.open(mode='w', newline='')
        metaLogWriter = CSVMetaLogFormatter(file=metaLog, record_collection=collection)
        metaLogWriter.write_csv()
        metaLog.close()

    # Test requires the metaLog and Summary file.
    def _check_metalog_csv_compatible(self):
        reader = CSVReader(attribute_names=['a', 'b', 'seed'], metric_names=['m_a', 'm_b'], discriminator='seed')
        metaLogResult = reader.check_compatible(self.metaLogPath)

        self.assertEqual(True, metaLogResult)

    # Test requires existing metaLog file.
    def _meta_logger_update_csv(self):
        meta_logger = MetaLogger(log_attributes=['a', 'b', 'seed'], log_metrics=['m_a', 'm_b'],
                                 discrimination_value='seed', log_file_path=self.metaLogPath)

        meta_logger.record_collection = self.load_new_records()
        meta_logger.save_logged_data()
        warnings.warn('Please check that {} is correct.'.format(self.metaLogPath.absolute()))

    # Test requires metaLog file.
    def _load_metalog_to_records(self):
        file = self.metaLogPath.open(mode='r', newline='')
        reader = CSVReader(attribute_names=['a', 'b', 'seed'], metric_names=['m_a', 'm_b'], discriminator='seed')
        result = reader.load_csv(file)
        expected = getTestRecords() + self.load_new_records().get_records()

        self.assertEqual(len(result), len(expected))

        file.close()

    def test_summary_monolithic(self):
        self.summaryPath = Path('data/test_summary.csv')
        self.metaLogPath = Path('data/test_meta_log.csv')
        self._write_summary()
        self._write_meta_log()
        self._check_csv_summary_compatible()
        self._summary_update_csv()
        self.summaryPath.unlink()
        self.metaLogPath.unlink()

    # Test creates the summary file.
    def _write_summary(self):
        collection = self.load_test_collection()

        summary = self.summaryPath.open(mode='w', newline='')
        summaryWriter = CSVSummaryLogFormatter(file=summary, record_collection=collection)
        summaryWriter.write_csv()
        summary.close()

    # Test requires the metaLog and Summary file.
    def _check_csv_summary_compatible(self):
        reader = CSVReader(attribute_names=['a', 'b', 'seed'], metric_names=['m_a', 'm_b'], discriminator='seed')
        summaryResult = reader.check_compatible(self.summaryPath)

        self.assertEqual(True, summaryResult)

    # Test requires existing summary file.
    def _summary_update_csv(self):
        summary_logger = SummaryLogger(log_attributes=['a', 'b', 'seed'], log_metrics=['m_a', 'm_b'],
                                       discrimination_value='seed', log_file_path=self.summaryPath,
                                       summary_log_file_path=self.summaryPath)

        summary_logger.record_collection = self.load_new_records()
        summary_logger.save_logged_data()
        warnings.warn('Please check that {} is correct.'.format(self.summaryPath.absolute()))
