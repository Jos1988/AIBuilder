from typing import List

import pandas as pd
from abc import ABC, abstractmethod

from AIBuilder import AI
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Printing import ConsolePrintStrategy, TesterPrinter, ReportPrintStrategy
from AIBuilder.Summizer import Summizer
from datetime import datetime
import hashlib


class Evaluator(ABC):

    def __init__(self):
        self.ml_model = None
        self.target_column = None
        self.result = {}

    def load_ml_model(self, ml_model: AI):
        self.ml_model = ml_model
        self.target_column = self.ml_model.get_evaluation_data().target_column_name
        self.result = {}

    @abstractmethod
    def evaluate(self) -> dict:
        pass


class BinaryClassificationEvaluator(Evaluator):

    def evaluate(self) -> dict:
        predictions = self.get_predictions()
        expected_labels = self.ml_model.get_evaluation_data().get_dataframe()[self.target_column]

        assert len(expected_labels) == len(predictions), 'Number of expectations does not match number of predictions.'

        evaluation_data_length = len(expected_labels)
        highest_frequencies = self.highest_frequency(expected_labels)
        successes = self.count_successful_predictions(expected_labels, predictions)

        self.result = {'bin class accuracy': successes / evaluation_data_length,
                       'bin class baseline': highest_frequencies / evaluation_data_length,
                       'bin class 1 - baseline': 1 - highest_frequencies / evaluation_data_length}

        return self.result

    @staticmethod
    def highest_frequency(expected_labels: pd.Series):
        class_counts = expected_labels.value_counts().to_dict()
        baseline_successes = 0
        for value, count in class_counts.items():
            if count > baseline_successes:
                baseline_successes = count

        return baseline_successes

    def get_predictions(self):
        predictor = self.ml_model.estimator.predict(input_fn=self.ml_model.evaluation_fn)
        predictions = self.load_predictions_from_predictor(predictor)
        return predictions

    @staticmethod
    def count_successful_predictions(expected_labels, predictions):
        success = 0
        for expected, prediction in zip(expected_labels, predictions):
            if expected == round(prediction):
                success += 1

        return success

    @staticmethod
    def load_predictions_from_predictor(predictor):
        predictions = []
        for p in predictor:
            # print(p)
            # regressor:
            # predictions.append(p['predictions'][0])
            # classifier:
            predictions.append(p['logistic'][0])
        return predictions


class AbstractAITester(ABC):

    @abstractmethod
    def train_AI(self):
        pass

    @abstractmethod
    def evaluate_AI(self):
        pass


class AITester(AbstractAITester):
    AI: AbstractAI

    def __init__(self, summizer: Summizer, evaluators: List[Evaluator] = None):
        self.summizer = summizer
        self.test_time = None
        self.results = {}
        self.console_print_strategy = ConsolePrintStrategy()
        self.console_printer = TesterPrinter(self.console_print_strategy)
        self.description_hash = None
        self.report_printer = None
        self.evaluators = []
        if evaluators is not None:
            self.evaluators = evaluators

    def logModelNotUnique(self):
        self.console_printer.line('')
        self.console_printer.separate()
        self.console_printer.line('AI already evaluated')
        self.print_description()
        self.console_printer.separate()

    def loadModel(self, ai):
        self.set_AI(ai)
        self.description_hash = self.stable_hash_description(ai.description)

    def set_report_printer(self):
        report = self.open_report_file('a')
        report_print_strategy = ReportPrintStrategy(report=report)

        self.report_printer = TesterPrinter(report_print_strategy)

    def set_AI(self, ai: AbstractAI):
        self.AI = ai
        self.set_report_printer()

    def is_unique(self) -> bool:
        report = self.open_report_file('r')
        for line in report:
            if -1 is not line.find(str(self.description_hash)):
                return False

        return True

    def train_AI(self):
        self.summizer.log('start training', None)
        self.determine_test_time()
        self.AI.train()
        self.summizer.log('finished training', None)

    def evaluate_AI(self):
        # todo move reporting logic to listener/observer
        self.summizer.log('start evaluation', None)
        self.results = self.AI.evaluate()
        self.run_evaluators()

        self.print_description()
        self.print_results()
        self.log_testing_report()
        self.report_printer.output.close_report()

        self.summizer.log('finished evaluation', None)
        self.summizer.summize(self.console_print_strategy)
        self.summizer.reset()

    def run_evaluators(self):
        for evaluator in self.evaluators:
            evaluator.load_ml_model(self.AI)
            evaluator_results = evaluator.evaluate()
            self.results.update(evaluator_results)

    def print_results(self):
        self.console_printer.separate()
        self.validate_results_set()
        self.console_printer.print_results(self.results)

    def print_description(self):
        self.console_printer.separate()
        self.console_printer.print_ai_description(
            ai=self.AI, time_stamp=self.test_time, ai_hash=self.description_hash)

    def log_testing_report(self):
        self.report_printer.line('')
        self.report_printer.print_ai_description(
            ai=self.AI, time_stamp=self.test_time, ai_hash=self.description_hash)
        self.report_printer.line('')
        self.report_printer.print_results(self.results)
        self.report_printer.separate()
        self.summizer.summize(self.report_printer.output)
        self.report_printer.separate()

    def validate_results_set(self):
        assert type(self.results) is dict, 'Test results not set in AI tester.'

    def determine_test_time(self):
        self.test_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    def validate_test_time(self):
        assert type(self.test_time) is str, 'Test time not set in AITester.'

    def open_report_file(self, mode: str):
        path = self.AI.get_log_dir() + '/' + self.AI.get_project_name() + '/ai_reports.txt'
        report = open(path, mode=mode)
        return report

    @staticmethod
    def stable_hash_description(description: dict):
        description = repr(description)
        description = description.encode('utf-8')
        hash_result = hashlib.md5(description)

        return hash_result.hexdigest()
