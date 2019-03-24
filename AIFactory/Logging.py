import csv
from typing import List

import numpy as np

from AIBuilder import AI


class LogRecord:
    values: dict

    def __init__(self, values: dict = None, discrimination_value: str = None):
        self.discrimination_value = discrimination_value
        if values is None:
            values = {}

        self.values = values

    def add_value(self, key, value):
        if key not in self.values:
            self.values[key] = value

    def get_group_values(self) -> dict:
        non_group_values = self.values.copy()
        non_group_values.pop(self.discrimination_value)

        return non_group_values

    def is_same_group(self, record):
        return self.get_group_values() == record.get_group_values()


class RecordCollection:
    record_groups: List[List[LogRecord]]

    def __init__(self, fieldnames: List[str]):
        self.fieldnames = fieldnames
        self.record_groups = []

    def is_same_format(self, record: LogRecord):
        record_field_keys = record.values.keys()

        return set(record_field_keys) == set(self.fieldnames)

    def add(self, record: LogRecord):
        assert self.is_same_format(record), 'Record has wrong format. record:{}, expecting: {}'.format(
            record.values.keys(), self.fieldnames)

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


class CSVConverter:

    def __init__(self, file, record_collection: RecordCollection):
        self.writer = csv.DictWriter(file, fieldnames=record_collection.fieldnames)
        self.record_collection = record_collection

    def writeRecord(self, record: LogRecord):
        self.writer.writerow(record.values)

    def writeCollection(self, record_collection: RecordCollection):
        for group in record_collection.record_groups:
            for record in group:
                self.writeRecord(record)

    def writeMetaLog(self):
        self.writer.writeheader()
        self.writeCollection(self.record_collection)


class MetaLogger:
    log_values: List[str]
    discrimination_value: str
    record_collection: RecordCollection

    def __init__(self, log_values: List[str], log_file_path: str, discrimination_value: str = 'seed'):
        self.discrimination_value = discrimination_value
        self.log_values = log_values
        self.log_file_path = log_file_path
        if discrimination_value not in log_values:
            self.log_values.append(discrimination_value)

        self.record_collection = RecordCollection(self.log_values)

    def log_ml_model(self, model: AI):
        assert hasattr(model, 'description'), 'not description set on model.'

        model_data = {}
        model_data.update(model.description)
        model_data.update(model.results)

        record = self.create_record_from_dict(model_data)
        self.record_collection.add(record)

    def create_record_from_dict(self, model_description: dict) -> LogRecord:
        record = LogRecord({}, discrimination_value=self.discrimination_value)
        for log_value in self.log_values:
            value = self.traverse_dict_for_value(model_description, log_value)
            assert len(value) is 1, 'multiple or no result found for "{}": {}'.format(log_value, value)
            value = value.pop()

            if type(value).__module__ is np.__name__:
                value = value.item()

            assert type(value) in [str, int, float, bool], \
                'can only store base data types in metalog, {} of type {} given'.format(value, type(value))

            record.add_value(log_value, value)

        return record

    def traverse_dict_for_value(self, data: dict, find_value: str):
        results = []
        assert type(data) is dict, 'data must be a dict, {} given.'.format(data)
        for key, value in data.items():
            if key is find_value and type(value) is not dict:
                results.append(value)

            if type(value) is dict:
                results = results + self.traverse_dict_for_value(value, find_value)

        return results

    def save_to_csv(self):
        file = open(self.log_file_path, mode='w', newline='')
        converter = CSVConverter(file, self.record_collection)
        converter.writeMetaLog()
        file.close()
