import os
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder


class NamingSchemeBuilder(Builder):

    def __init__(self):
        self.versions = []
        self.AI = None
        self.existing_names = []

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.NAMING_SCHEME

    def validate(self):
        pass

    def build(self, neural_net: AbstractAI):
        self.AI = neural_net
        self.existing_names = self.get_logged_names()

        if self.AI.get_name() is None or self.AI.get_name() is self.AI.get_project_name():
            self.generate_name()
            return

        if self.AI.get_name() in self.existing_names:
            self.AI.set_name(self.AI.get_name() + '_1')
            return

        if self.AI.get_name() is not None:
            return

        raise RuntimeError('Naming scheme failed to set name.')

    def generate_name(self):
        for name in self.existing_names:
            version = self.get_version(name=name)
            if version is not False:
                self.versions.append(version)

        last_version = 0
        if len(self.versions) > 0:
            last_version = max(self.versions)

        new_version = last_version + 1
        name = self.AI.get_project_name() + '_' + str(new_version)
        self.AI.set_name(name)

    def get_logged_names(self):
        tensor_board_path = self.AI.get_log_dir() + '/' + self.AI.get_project_name() + '/tensor_board'
        return next(os.walk(tensor_board_path))[1]

    def get_version(self, name: str):
        exploded = name.split('_')

        if exploded[0] == self.AI.get_project_name() and len(exploded) > 1 and exploded[1].isnumeric():
            return int(exploded[1])

        return False
