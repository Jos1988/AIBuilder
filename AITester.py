from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.Summizer import Summizer
from unittest import TestCase, mock


# AbstractAITester
class AbstractAITester(ABC):

    @abstractmethod
    def train_AI(self):
        pass

    @abstractmethod
    def evaluate_AI(self):
        pass


class AITester(AbstractAITester):
    AI: AbstractAI

    def __init__(self, summizer: Summizer, model_dir: str):
        self.summizer = summizer
        self.model_dir = model_dir
        self.results = {}

    def set_AI(self, ai: AbstractAI):
        self.AI = ai

    def train_AI(self):
        self.summizer.log('start training', None)
        self.AI.train()
        self.summizer.log('finished training', None)

    def evaluate_AI(self):
        self.summizer.log('start evaluation', None)

        # todo find some way to get specs for ai.
        # self.generateModelDirName(self.AI.getAIdetails())
        # self.createReport()
        # self.storeReport()

        self.results = self.AI.evaluate(model_dir=model_dir)
        self.print_evaluation_results()
        self.summizer.log('finished evaluation', None)
        self.summizer.summize()

    def print_evaluation_results(self):
        for label, value in self.results.items():
            print(label + ': ' + str(value))


class AITesterTest(TestCase):

    def setUp(self):
        self.ai = mock.Mock('AITester.AbstractAI')
        self.ai.train = mock.Mock(name='train')

        summizer = mock.patch('AITester.Summizer')
        summizer.log = mock.Mock(name='log')

        dir_path = 'path/'
        self.ai_tester = AITester(summizer=summizer, model_dir=dir_path)
        self.ai_tester.AI = self.ai

    def test_training(self):
        self.ai_tester.train_AI()
        self.ai.train.assert_called()
        self.assertEqual(2, self.ai_tester.summizer.log.call_count)

    def test_evaluation(self):
        pass

    def test_print_results(self):
        pass

    def test_generate_model_dir_name(self):
        pass

    def test_create_report(self):
        pass

    def test_store_report(self):
        pass
