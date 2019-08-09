import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np

from AIBuilder import AI
from AIBuilder.AI import AbstractAI

BASE_TYPES = [str, int, float, bool]


class LogRecord:
    """ Record of one ml model that has been trained an evaluated.
    """

    attributes: dict
    metrics: dict
    grouping_values: dict

    def __init__(self, attributes: dict, metrics: dict = None, discriminator_name: str = None):

        self.attributes = {}
        if attributes is not None:
            for attr_name, attr_value in attributes.items():
                self.add_attribute(attr_name, attr_value)

        self.metrics = {}
        if metrics is not None:
            for metr_name, metr_value in metrics.items():
                self.add_metric(metr_name, metr_value)

        self.discriminator_name = discriminator_name
        if metrics is None:
            self.metrics = {}

    def add_attribute(self, key, value):
        if key not in self.attributes:
            self.attributes[str(key)] = str(value)

    def add_metric(self, key, value):
        if key not in self.metrics:
            self.metrics[str(key)] = str(value)

    def get_all_values(self):
        all_values = self.attributes.copy()
        all_values.update(self.metrics)

        return all_values

    def get_all_value_names(self):
        return list(self.attributes.keys()) + list(self.metrics.keys())

    def get_group_values(self) -> dict:
        non_group_values = self.attributes.copy()
        non_group_values.pop(self.discriminator_name)

        return non_group_values

    def is_same_group(self, record):
        """ Check if record attributes are equal. If so they should be place in the same group in the respective
        collection as they represent results from the same architecture.
        """
        return self.get_group_values() == record.get_group_values()


class RecordCollection:
    """ Collection of records organize in groups, a group represents ml models with the same architecture (attributes).
    This way we can observe performance of the architecture across different instances.
    Models in the same group should differ by discriminator. The discriminator functions as a 'seed' adding some
    aspect of random variation to the model.
    """
    record_groups: List[List[LogRecord]]

    def __init__(self, log_attributes: List[str], log_metrics: List[str], discriminator: str):
        self.log_attributes = log_attributes
        self.log_metrics = log_metrics
        self.all_values = log_attributes + log_metrics
        self.discriminator = discriminator
        self.record_groups = []

    def get_records(self) -> List[LogRecord]:
        """ Return list of all records from all groups
        """
        records = []
        for group in self.record_groups:
            records += group

        return records

    def is_same_format(self, record: LogRecord):
        """ Check if record fit the format of the collection, meaning, it has the same attributes, discriminator and
        metrics.
        """
        return set(record.get_all_value_names()) == set(self.all_values)

    def add(self, record: LogRecord):
        assert self.is_same_format(record), 'Record has wrong format. got {}, expecting: {}'.format(
            record.get_all_value_names(), self.all_values)

        for group in self.record_groups:
            if len(group) == 0:
                continue

            group_sample = group[0]
            if group_sample.is_same_group(record):
                group.append(record)
                return

        self.record_groups.append([record])

    def has(self, record: LogRecord):
        for group in self.record_groups:
            if record in group:
                return True

        return False

    def remove(self, record: LogRecord):
        if not self.has(record):
            return

        for group in self.record_groups:
            if record in group:
                group.remove(record)

            if 0 == len(group):
                self.record_groups.remove(group)

    def clear(self):
        self.record_groups = []

    def __len__(self):
        length = 0
        for group in self.record_groups:
            length += len(group)

        return length


class CSVReader:
    attribute_names: List[str]
    metric_names: List[str]
    discriminator: str

    def __init__(self, attribute_names: List[str], metric_names: List[str], discriminator: str):
        self.attribute_names = attribute_names
        self.metric_names = metric_names
        self.all_names = metric_names + attribute_names
        self.discriminator = discriminator

    def check_compatible(self, file_path: Path) -> bool:
        file = file_path.open(mode='r', newline='')
        csv_reader = csv.DictReader(file)

        if None is csv_reader.fieldnames or 0 is len(csv_reader.fieldnames):
            return False

        missing_columns = (set(self.all_names) - set(csv_reader.fieldnames))
        file.close()

        return len(missing_columns) is 0

    def load_csv(self, file) -> RecordCollection:
        """ Load CSV log file and returns respective records."""
        csv_reader = csv.DictReader(file)
        loaded_records = RecordCollection(self.attribute_names, self.metric_names, self.discriminator)
        for csv_record in csv_reader:
            if not self.is_row_record(csv_record):
                continue

            record = self.load_record(csv_record)
            loaded_records.add(record)

        file.close()
        return loaded_records

    @staticmethod
    def is_row_record(values: dict) -> bool:
        for key, value in values.items():
            if value is '':
                return False

        return True

    def load_record(self, csv_record):
        attributes = {}
        metrics = {}
        for field_name, field_value in csv_record.items():
            if field_name in self.attribute_names:
                attributes[field_name] = field_value
            else:
                metrics[field_name] = field_value

        return LogRecord(attributes=attributes, metrics=metrics, discriminator_name=self.discriminator)


class CSVFormatter(ABC):

    def __init__(self, file, record_collection: RecordCollection):
        """ Writes a collection to a csv file. """
        self.writer = csv.DictWriter(file, fieldnames=record_collection.all_values)
        self.record_collection = record_collection

    @staticmethod
    def generate_group_summary(group: List[LogRecord], write_attributes: bool) -> dict:
        summary_data_keys = group[0].get_all_value_names()
        discriminator = group[0].discriminator_name
        attribute_names = group[0].attributes.keys()

        summary = {}
        for label in summary_data_keys:
            if discriminator is not None and label == discriminator:
                summary[label] = 'n={}'.format(len(group))
                continue

            if False is write_attributes and label in attribute_names:
                summary[label] = ''
                continue

            if label in attribute_names:
                summary[label] = group[0].attributes[label]
                continue

            all_label_values = []
            for record in group:
                all_label_values.append(float(record.metrics[label]))

            average = sum(all_label_values) / len(all_label_values)
            summary[label] = str(average)
        return summary

    @staticmethod
    def empty_dict_values(data: dict):
        empty_dict = {}
        for key, value in data.items():
            empty_dict[key] = ''

        return empty_dict

    def write_record(self, record: LogRecord):
        self.writer.writerow(record.get_all_values())

    def write_empty_row(self, record: LogRecord):
        empty_values = self.empty_dict_values(record.get_all_values())
        self.writer.writerow(empty_values)

    def write_group_summary(self, group: List[LogRecord], write_attributes: bool = False):
        summary = self.generate_group_summary(group, write_attributes)
        self.writer.writerow(summary)

    @abstractmethod
    def write_csv(self):
        pass


class CSVMetaLogFormatter(CSVFormatter):
    def write_collection(self, record_collection: RecordCollection):
        for group in record_collection.record_groups:
            for record in group:
                self.write_record(record)
            self.write_group_summary(group)
            self.write_empty_row(group[0])

    def write_csv(self):
        self.writer.writeheader()
        self.write_collection(self.record_collection)


class CSVSummaryLogFormatter(CSVFormatter):
    def write_collection(self, record_collection: RecordCollection):
        for group in record_collection.record_groups:
            self.write_group_summary(group, write_attributes=True)

    def write_csv(self):
        self.writer.writeheader()
        self.write_collection(self.record_collection)


class CSVLogger(ABC):
    log_attributes: List[str]
    discrimination_value: str
    record_collection: RecordCollection

    def __init__(self, log_attributes: List[str], log_metrics: List[str], discrimination_value: str,
                 log_file_path: Path):
        self.log_attributes = log_attributes
        self.log_metrics = log_metrics
        self.discrimination_value = discrimination_value

        self.log_file_path = log_file_path

        self.collection_fields = log_attributes + log_metrics
        if discrimination_value not in log_attributes:
            self.log_attributes.append(discrimination_value)

        self.record_collection = RecordCollection(self.log_attributes, self.log_metrics, self.discrimination_value)

    def add_ml_model(self, model: AI):
        assert hasattr(model, 'description'), 'not description set on model.'

        model_data = {}
        model_data.update(model.description)
        model_data.update(model.results)

        record = self.create_record_from_dict(model_description=model.description, metrics=model.results)

        self.record_collection.add(record)

    def create_record_from_dict(self, model_description: dict, metrics: dict) -> LogRecord:
        record = LogRecord({}, discriminator_name=self.discrimination_value)
        self.load_attributes(model_description, record)
        self.load_metrics(metrics, record)

        return record

    def load_attributes(self, model_description: dict, record: LogRecord):
        for log_value in self.log_attributes:
            value = self.traverse_dict_for_value(model_description, log_value)
            value = self.check_value(log_value, value)

            record.add_attribute(log_value, value)

    def load_metrics(self, metrics: dict, record: LogRecord):
        for log_value in self.log_metrics:
            value = self.traverse_dict_for_value(metrics, log_value)
            value = self.check_value(log_value, value)

            record.add_metric(log_value, value)

    @staticmethod
    def check_value(log_value, value):
        assert len(value) is 1, f'multiple or no result found for "{log_value}", results: {value}.'
        value = value.pop()

        if type(value).__module__ is np.__name__:
            value = value.item()

        if type(value) is list:
            value = CSVLogger.try_flatten_list(value)

        CSVLogger.check_base_types(value)

        return value

    @staticmethod
    def check_base_types(value):
        assert type(value) in BASE_TYPES, \
            f'can only store base data types in metalog, {value} of type {type(value)} given.'

    def traverse_dict_for_value(self, data: dict, find_value: str):
        results = []
        assert type(data) is dict, f'data must be a dict, {type(data)} given.'
        for key, value in data.items():
            if key == find_value and type(value) is not dict:
                results.append(value)
                continue

            if type(value) is dict:
                results = results + self.traverse_dict_for_value(value, find_value)

        return results

    @staticmethod
    def try_flatten_list(input: list) -> str:
        result = ''
        for element in input:
            if type(element) is list:
                result = result + ' ' + CSVLogger.try_flatten_list(element)
                continue

            CSVLogger.check_base_types(element)
            result = result + ' ' + str(element)

        return result

    @abstractmethod
    def save_logged_data(self):
        pass

    def Load_records_from_log(self, records_to_save: RecordCollection):
        reader = CSVReader(self.log_attributes, self.log_metrics, self.discrimination_value)

        assert reader.check_compatible(self.log_file_path)

        file = self.log_file_path.open(mode='r', newline='')
        existing_records = reader.load_csv(file)

        for record in existing_records.get_records():
            records_to_save.add(record)

        file.close()


class MetaLogger(CSVLogger):

    def save_logged_data(self):
        records_to_save = self.record_collection

        if 0 is len(records_to_save):
            return

        if self.log_file_path.is_file():
            self.Load_records_from_log(records_to_save)
            self.log_file_path.unlink()

        file = self.log_file_path.open(mode='w', newline='')

        converter = CSVMetaLogFormatter(file, records_to_save)
        converter.write_csv()

        self.record_collection.clear()
        file.close()


class SummaryLogger(CSVLogger):

    def __init__(self, log_attributes: List[str], log_metrics: List[str], discrimination_value: str,
                 log_file_path: Path, summary_log_file_path: Path):

        super().__init__(log_attributes, log_metrics, discrimination_value, log_file_path)

        self.summary_log_file_path = summary_log_file_path

    def save_logged_data(self):
        records_to_save = self.record_collection

        if 0 is len(records_to_save):
            return

        if self.log_file_path.is_file():
            self.Load_records_from_log(records_to_save)

        file = self.summary_log_file_path.open(mode='w', newline='')
        converter = CSVSummaryLogFormatter(file, records_to_save)
        converter.write_csv()
        self.record_collection.clear()
        file.close()
