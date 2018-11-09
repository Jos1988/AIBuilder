from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.Summizer import Summizer


# AbstractAITester
class AbstractAITester(ABC):

    @abstractmethod
    def train_AI(self):
        pass

    @abstractmethod
    def evaluate_AI(self):
        pass


class AITester(AbstractAITester):

    def __init__(self, ai: AbstractAI, summizer: Summizer, model_dir: str):
        self.AI = ai
        self.summizer = summizer
        self.model_dir = model_dir
        self.results = {}

    def train_AI(self):
        self.summizer.log('start training', None)
        self.AI.train()
        self.summizer.log('finished training', None)

    def evaluate_AI(self):
        self.summizer.log('start evaluation', None)

        # todo find some way to get specs for ai.
        # self.generateModelDirName(self.AI.getAIdetails())

        self.results = self.AI.evaluate(model_dir=model_dir)
        self.print_evaluation_results()
        self.summizer.log('finished evaluation', None)
        self.summizer.summize()

    def print_evaluation_results(self):
        for label, value in self.results.items():
            print(label + ': ' + str(value))
