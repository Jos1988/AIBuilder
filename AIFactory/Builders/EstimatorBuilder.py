from AIBuilder.AIFactory.Specifications.BasicSpecifications import TypeSpecification
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
import tensorflow as tf


class EstimatorBuilder(Builder):
    LINEAR_REGRESSOR = 'linear_regressor'
    valid_estimator_types = [LINEAR_REGRESSOR]
    estimator: str
    estimator_type = None

    def __init__(self, estimator_type: str):
        self.set_estimator(estimator_type)

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.DATA_MODEL, self.NAMING_SCHEME]

    @property
    def builder_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.estimator_type = TypeSpecification('estimator_type', estimator_type, self.valid_estimator_types)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        if self.estimator_type() is self.LINEAR_REGRESSOR:
            estimator = tf.estimator.LinearRegressor(
                feature_columns=neural_net.training_data.get_tf_feature_columns(),
                optimizer=neural_net.optimizer,
                model_dir=self.render_model_dir(neural_net)
            )

            neural_net.set_estimator(estimator)
            return

        raise RuntimeError('Estimator Builder failed to set estimator.')

    @staticmethod
    def render_model_dir(ai) -> str:
        return ai.get_log_dir() + '/' + ai.get_project_name() + '/tensor_board/' + ai.get_name()
