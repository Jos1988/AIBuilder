from typing import List

import pandas as pd
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from AIBuilder import AI
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Printing import PrintStrategy
from AIBuilder.Summizer import Summizer
from datetime import datetime
import hashlib


class Evaluator(ABC):
    ml_model: AI

    def __init__(self):
        self.ml_model = None
        self.target_column = None
        self.result = {}

    def load_ml_model(self, ml_model: AI):
        self.ml_model = ml_model
        self.target_column = self.ml_model.get_evaluation_data().target_column_name
        self.result = {}

    @abstractmethod
    def do_run(self, ml_model: AI) -> bool:
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        pass


class GainBasedFeatureImportance(Evaluator):
    """
    Relevant for Boosted Trees, calculates gain from features and loads them into a 2-D list.
    tensorflos docs: https://www.tensorflow.org/tutorials/estimators/boosted_trees_model_understanding#1_gain-based_feature_importances
    example:
    { 'feature importance': [['feature1': 3.5]
                              ['feature2': 1.3]
                              ['feature3': 0.2]]}
    """

    def do_run(self, ml_model: AI) -> bool:
        return isinstance(ml_model.estimator, tf.estimator.BoostedTreesClassifier)

    def evaluate(self) -> dict:
        estimator = self.ml_model.estimator
        assert isinstance(estimator, tf.estimator.BoostedTreesClassifier)

        importance = estimator.experimental_feature_importances(normalize=False)
        importance = np.column_stack([importance[0], importance[1]])

        return {'feature importance': importance}


class AccuracyBaselineDiff(Evaluator):
    """
    Adds the difference between accuracy and the accuracy baseline to the eval metrics.
    { 'accuracy_diff':0.015}
    """

    CLASSIFIERS = [
        tf.estimator.BoostedTreesClassifier,
        tf.estimator.DNNClassifier,
        tf.estimator.LinearClassifier,
    ]

    def do_run(self, ml_model: AI) -> bool:
        """ Has to be a classifier, because 'accuracy' and  'accuracy_baseline' are required metrics.
        """

        for estimator in self.CLASSIFIERS:
            if isinstance(ml_model.estimator, estimator):
                return True

        return False

    def evaluate(self) -> dict:
        accuracy = float(self.ml_model.results['accuracy'])
        accuracyBaseline = float(self.ml_model.results['accuracy_baseline'])

        return {'accuracy_diff': accuracy - accuracyBaseline}


class BinaryClassificationEvaluator(Evaluator):

    def do_run(self, ml_model: AI) -> bool:
        return False

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
    ml_model: AbstractAI

    def __init__(self, summizer: Summizer, evaluators: List[Evaluator] = None):
        self.summizer = summizer
        self.test_time = None
        self.description_hash = None
        self.report_printer = None
        self.evaluators = []
        if evaluators is not None:
            self.evaluators = evaluators

    def loadModel(self, ai):
        self.set_ml_model(ai)
        self.description_hash = self.stable_hash_description(ai.description)

    def set_ml_model(self, ml_model: AbstractAI):
        self.ml_model = ml_model

    def is_unique(self) -> bool:
        report = self.get_report_file('r')
        for line in report:
            if -1 is not line.find(str(self.description_hash)):
                report.close()
                return False

        report.close()
        return True

    def train_AI(self):
        self.summizer.log('start training', None)
        self.determine_test_time()
        self.ml_model.train()
        self.summizer.log('finished training', None)

    def evaluate_AI(self):
        self.summizer.log('start evaluation', None)
        self.ml_model.evaluate()
        self.run_evaluators()
        self.summizer.log('finished evaluation', None)

    def run_evaluators(self):
        for evaluator in self.evaluators:
            if not evaluator.do_run(self.ml_model):
                continue

            evaluator.load_ml_model(self.ml_model)
            evaluator_results = evaluator.evaluate()
            self.ml_model.results.update(evaluator_results)

    def summizeTime(self, print_strategy: PrintStrategy):
        self.summizer.summize(print_strategy)

    def validate_results_set(self):
        assert type(self.ml_model.results) is dict, 'Test results not set in AI tester.'

    def determine_test_time(self):
        self.test_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    def validate_test_time(self):
        assert type(self.test_time) is str, 'Test time not set in AITester.'

    def get_report_file(self, mode: str):
        path = self.ml_model.get_log_dir() + '/' + self.ml_model.get_project_name() + '/ai_reports.txt'
        report = open(path, mode=mode)
        return report

    @staticmethod
    def stable_hash_description(description: dict):
        description = repr(description)
        description = description.encode('utf-8')
        hash_result = hashlib.md5(description)

        return hash_result.hexdigest()


class Predictor:

    def __init__(self, categories: List[str]):
        self.categories = categories

    def predict(self, models: List[AI.AI]) -> List:
        predictions = self.load_predictions(models)

        final_predictions = []
        for item_predictions in predictions:
            grouped_probabilities = self.group_probabilities_by_class(item_predictions)
            averaged_prediction = self.average_grouped_prediction(grouped_probabilities)
            final_predictions.append(averaged_prediction)

        final_predictions = self.load_predicted_categories(final_predictions)

        return final_predictions

    def load_predictions(self, models: List[AI.AI]) -> np.array:
        """ Loads 2D array of predictions where the second dimension equals the number of models and the first
            the number of predictions.

        ex: [
            [prediction1, prediction2, prediction3 ..., predictionN], #item1
            [prediction1, prediction2, prediction3 ..., predictionN], #item2
            ...
            [prediction1, prediction2, prediction3 ..., predictionN], #itemN
        ]
        """
        predictions = []
        length = 0
        for model in models:
            gen = model.predict()
            model_predictions = [item_prediction for item_prediction in gen]

            if length is 0:
                length = len(model_predictions)
            else:
                assert length == len(model_predictions), 'Expected {} predictions got {}'\
                    .format(length, len(model_predictions))

            predictions.append(model_predictions)

        predictions = np.array(predictions)
        prediction_formatted = [predictions[:, i] for i in range(0, length)]

        return prediction_formatted

    def prediction_to_category(self, predictions: List[float]):
        highest_prediction = max(predictions)
        i = 0
        for prediction in predictions:
            if prediction == highest_prediction:
                return self.categories[i]
            i += 1

    def load_predicted_categories(self, final_predictions):
        final_predictions = [self.prediction_to_category(predictions) for predictions in final_predictions]
        return final_predictions

    def average_grouped_prediction(self, grouped_probabilities: dict):
        """ Averages prediction per class.

        :param grouped_probabilities: dict
            ex: {'0': [0.1, 0.2, 0.3], '1': [0.9, 0.8, 0.7}

        :return: returns returns Averaged dict:
            ex: {'0': 0.2, '1': 0.8}
        """

        averaged_prediction = [sum(pred) / len(pred) for index, pred in grouped_probabilities.items()]
        return averaged_prediction

    def group_probabilities_by_class(self, item_predictions: dict):
        """ Groups predictions from by class
        item_predictions: List of dicts from tf.estimator.predict()

        returns dict of predictions for each category
                {'0': [0.1, 0.2, 0.1], '1': [0.9, 0.8, 0.9}
        """
        avg_predictions = {}
        for prediction in item_predictions:
            assert 'probabilities' in prediction
            probabilities = prediction['probabilities']

            i = 0
            for probability in probabilities:
                if i not in avg_predictions:
                    avg_predictions[i] = []

                avg_predictions[i].append(probability)
                i += 1
        return avg_predictions
