from AIBuilder.AI import AI, AbstractAI
from AIBuilder.AIFactory.Builders import Builder


# todo: set all builders in some kind of recipe and allow cloning and modding to variate the recipes.
#  Or rather, set a default on the factory, possibly from another file and create a 'rotate builders' method.
#  Where the factory is given an numer of new builders and cycles through all possible combinations.

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
        self.validate_builders(builders)
        artificial_intelligence = AI(self.project_name, self.log_dir, self.ai_name)

        self.sortBuilders(builders)

        ai_description = {}
        for builder in self.builders_sorted:
            builder.build(artificial_intelligence)

            builder_description = builder.describe()
            ai_description[builder.builder_type] = builder_description

        artificial_intelligence.description = ai_description

        return artificial_intelligence

    @staticmethod
    def validate_builders(builders: list):
        for builder in builders:
            builder.validate()

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
