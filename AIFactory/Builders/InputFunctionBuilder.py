import unittest
from unittest import mock
from AIBuilder.AIFactory.Specifications.BasicSpecifications import TypeSpecification, DataTypeSpecification
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
import AIBuilder.InputFunctionHolder as InputFunctionHolder


class InputFunctionBuilder(Builder):
    BASE_FN = 'base_fn'

    VALID_FN_NAMES = [BASE_FN]

    def __init__(self, train_fn: str, train_kwargs: dict, evaluation_fn: str, evaluation_kwargs: dict):
        self.train_fn_name = TypeSpecification('test_dir function', train_fn, self.VALID_FN_NAMES)
        self.train_kwargs = DataTypeSpecification('train_kwargs', train_kwargs, dict)

        self.evaluation_fn_name = TypeSpecification('evaluation function', evaluation_fn, self.VALID_FN_NAMES)
        self.evaluation_kwargs = DataTypeSpecification('evaluation_kwargs', evaluation_kwargs, dict)

        self.fn_holder = InputFunctionHolder

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.INPUT_FUNCTION

    def assign_fn(self, fn_name: str, kwargs: dict):
        assert hasattr(self.fn_holder, fn_name), 'Function {} not known in function holder.'.format(fn_name)
        assert callable(getattr(self.fn_holder, fn_name)), 'Function {} is not callable.'.format(fn_name)
        fn = getattr(self.fn_holder, fn_name)

        return lambda: fn(**kwargs)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        self.train_kwargs.value['data_model'] = neural_net.training_data
        self.evaluation_kwargs.value['data_model'] = neural_net.evaluation_data

        train_function = self.assign_fn(self.train_fn_name(), self.train_kwargs())
        evaluation_function = self.assign_fn(self.evaluation_fn_name(), self.evaluation_kwargs())

        neural_net.set_training_fn(train_function)
        neural_net.set_evaluation_fn(evaluation_function)


class TestEstimatorBuilder(unittest.TestCase):

    def setUp(self):
        self.input_fn_builder = InputFunctionBuilder(train_fn=InputFunctionBuilder.BASE_FN,
                                                     train_kwargs={'batch_size': 100, 'epoch': 1},
                                                     evaluation_fn=InputFunctionBuilder.BASE_FN,
                                                     evaluation_kwargs={'batch_size': 100, 'epoch': 1})

    def test_validate(self):
        self.input_fn_builder.validate()

    def test_build(self):
        train_data = mock.Mock()
        eval_data = mock.Mock()

        arti = mock.Mock()
        arti.training_data = train_data
        arti.evaluation_data = eval_data

        arti.set_training_fn = mock.Mock()
        arti.set_evaluation_fn = mock.Mock()

        self.input_fn_builder.build(arti)

        arti.set_training_fn.assert_called_once()
        arti.set_evaluation_fn.assert_called_once()


if __name__ == '__main__':
    unittest.main()
