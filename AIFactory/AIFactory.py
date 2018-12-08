from typing import List, Dict, Any
from AIBuilder.AI import AI, AbstractAI
from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.AIFactory.Printing import ConsolePrintStrategy, FactoryPrinter


class BuilderSorter(object):
    builders_sorted = list
    loaded_builders = list
    builders_by_name = dict
    unloaded_builders = list

    def clear(self):
        self.builders_sorted = []
        self.loaded_builders = []
        self.builders_by_name = {}
        self.unloaded_builders = []

    def sort(self, builders: list) -> list:
        self.clear()

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

        return self.builders_sorted

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
            assert dependency in self.builders_by_name, '{} has unknown dependency: {}' \
                .format(builder.__class__.__name__, dependency)

            dependent_builder = self.builders_by_name[dependency]

            if self.has_unloaded_dependencies(dependent_builder):
                return self.get_next_loadable_dependency(dependent_builder)

            if dependent_builder in self.unloaded_builders:
                return dependent_builder

            continue


class PermutationGenerator(object):
    permutations: List[List[Any]]
    builder_types: List[str]
    grouped_builders: Dict[str, List[Builder]]
    all_builders: List[Builder]

    def generate(self, builders: List[Builder]) -> list:
        self.all_builders = builders
        self.group_builders()
        self.walk_layers()

        return self.permutations

    def group_builders(self):
        self.grouped_builders = {}
        self.builder_types = []
        for builder in self.all_builders:
            if builder.builder_type not in self.grouped_builders:
                self.grouped_builders[builder.builder_type] = []
                self.builder_types.append(builder.builder_type)

            self.grouped_builders[builder.builder_type].append(builder)

    def walk_layers(self):
        layers = self.builder_types
        self.permutations = [[]]
        for layer in layers:
            self.permutations = self.walk_layer_with_permutations(layer, permutations=self.permutations)

    def walk_layer_with_permutations(self, layer_name: str, permutations: List[list]):
        layer_options = self.grouped_builders[layer_name]

        new_permutations = []
        for permutation in permutations:
            for option in layer_options:
                new_permutations.append(permutation + [option])

        return new_permutations


class AIFactory:

    builder_permutations: List[List[Builder]]

    def __init__(self, builders: List[Builder],  project_name: str, log_dir: str):
        console_strategy = ConsolePrintStrategy()
        self.console_printer = FactoryPrinter(console_strategy)
        self.sorter = BuilderSorter()
        self.project_name = project_name
        self.log_dir = log_dir
        self.permutation_generator = PermutationGenerator()
        self.builder_permutations = self.permutation_generator.generate(builders=builders)

    def count_remaining_ai(self):
        return len(self.builder_permutations)

    def has_next_ai(self):
        return self.count_remaining_ai() is not 0

    def create_next_ai(self):
        next_builder_permutation = self.builder_permutations.pop()

        return self.create_AI(next_builder_permutation)

    def create_AI(self, builders: list, ai_name: str = None) -> AbstractAI:
        self.validate_builders(builders)
        artificial_intelligence = AI(self.project_name, self.log_dir, ai_name)

        builders_sorted = self.sorter.sort(builders)

        ai_description = {}
        for builder in builders_sorted:
            builder.build(artificial_intelligence)

            builder_description = builder.describe()
            ai_description[builder.builder_type] = builder_description

        artificial_intelligence.description = ai_description

        return artificial_intelligence

    @staticmethod
    def validate_builders(builders: list):
        for builder in builders:
            builder.validate()

    def print_remaining_ai(self):
        count = self.count_remaining_ai()
        self.console_printer.print_remaining_ai(count)
