from AIBuilder.AI import AI, AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
from AIBuilder.AIFactory.Builders.DataBuilder import DataBuilder
from AIBuilder.AIFactory.Builders.EstimatorBuilder import EstimatorBuilder
from AIBuilder.AIFactory.Builders.OptimizerBuilder import OptimizerBuilder
from AIBuilder.AIFactory.Builders.ScrubAdapter import ScrubAdapter
from AIBuilder.Data import MetaData
import AIBuilder.DataScrubbing as scrubber
import unittest


class AIFactory:

    def __init__(self, project_name: str, log_dir: str, ai_name: str = None):
        self.project_name = project_name
        self.log_dir = log_dir
        self.ai_name = ai_name
        self.builders_by_name = {}
        self.loaded_builders = []
        self.unloaded_builders = []

        self.builders_sorted = []

    def create_AI(self, builders: list) -> AbstractAI:
        artificial_intelligence = AI(self.project_name, self.log_dir, self.ai_name)

        for builder in builders:
            builder.validate()

        self.sortBuilders(builders)

        ai_description = {}
        for builder in self.builders_sorted:
            builder.build(artificial_intelligence)

            builder_description = builder.describe()
            ai_description[builder.builder_type] = builder_description

        artificial_intelligence.description = ai_description

        return artificial_intelligence

    def sortBuilders(self, builders: list):
        for builder in builders:
            self.unloaded_builders.append(builder)
            self.builders_by_name[builder.builder_type] = builder

        while len(self.unloaded_builders) is not 0:
            builder = self.unloaded_builders.pop()
            if not self.has_unloaded_dependencies(builder):
                self.load_builder(builder)
                continue

            self.unloaded_builders.append(builder)

            dependency = self.get_next_loadable_dependency(builder)
            self.unloaded_builders.remove(dependency)
            self.load_builder(dependency)

    def has_unloaded_dependencies(self, builder: Builder):
        dependencies = builder.dependent_on
        if len(dependencies) is 0:
            return False

        for dependency in dependencies:
            if dependency not in self.loaded_builders:
                return True

        return False

    def load_builder(self, builder: Builder):
        self.loaded_builders.append(builder.builder_type)
        if builder in self.unloaded_builders:
            self.unloaded_builders.remove(builder)

        self.builders_sorted.append(builder)

    def get_next_loadable_dependency(self, builder: Builder) -> Builder:
        dependencies = builder.dependent_on

        if len(dependencies) == 0:
            raise RuntimeError('{} has no dependencies, so cannot get next loadable dependency.'
                               .format(builder.__class__.__name__))

        for dependency in dependencies:
            assert dependency in self.builders_by_name, '{} has unknown dependency: {}'\
                .format(builder.__class__.__name__, dependency)

            dependent_builder = self.builders_by_name[dependency]

            if self.has_unloaded_dependencies(dependent_builder):
                return self.get_next_loadable_dependency(dependent_builder)

            if dependent_builder in self.unloaded_builders:
                return dependent_builder

            continue


class TestAIFactory(unittest.TestCase):

    def setUp(self):
        self.factory = AIFactory()

    def test_create_AI(self):
        metadata = MetaData()
        metadata.define_numerical_columns(['feature_2', 'feature_3', 'target_1'])
        metadata.define_categorical_columns(['feature_1'])

        data_builder = DataBuilder(data_source='../data/test_data.csv', target_column='target_1',
                                   validation_data_percentage=20,
                                   feature_columns={
                                       'feature_1': DataBuilder.CATEGORICAL_COLUMN_VOC_LIST,
                                       'feature_2': DataBuilder.NUMERICAL_COLUMN,
                                       'feature_3': DataBuilder.NUMERICAL_COLUMN
                                   },
                                   metadata=metadata)

        estimator_builder = EstimatorBuilder(estimator_type=EstimatorBuilder.LINEAR_REGRESSOR)
        optimizer_builder = OptimizerBuilder(optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
                                             learning_rate=0.00002,
                                             gradient_clipping=5.0)

        missing_data_scrubber = scrubber.MissingDataScrubber('missing data')
        average_scrubber = scrubber.AverageColumnScrubber(('feature_2', 'feature_3'), 'feature_4')
        scrub_adapter = ScrubAdapter()
        scrub_adapter.add_scrubber(missing_data_scrubber)
        scrub_adapter.add_scrubber(average_scrubber)

        artie = self.factory.create_AI([data_builder, estimator_builder, optimizer_builder, scrub_adapter])
        self.assertIsNotNone(artie.optimizer)
        self.assertIsNotNone(artie.estimator)
        self.assertIsNotNone(artie.training_data)
        self.assertIsNotNone(artie.validation_data)


if __name__ == '__main__':
    unittest.main()
