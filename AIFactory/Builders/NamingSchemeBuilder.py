import os
import unittest
from unittest import mock
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


class TestNamingScheme(unittest.TestCase):

    def setUp(self):
        self.naming_scheme = NamingSchemeBuilder()
        self.arti = mock.Mock('NamingSchemeBuilder.AIBuilder.AI')
        self.arti.get_log_dir = mock.Mock()
        self.arti.get_log_dir.return_value = '../../../builder projects/log'
        self.arti.get_project_name = mock.Mock()
        self.arti.set_name = mock.Mock()
        self.arti.get_name = mock.Mock()
        self.arti.get_name.return_value = None

    @mock.patch('NamingSchemeBuilder.os.walk')
    def test_generate_name(self, walk):
        walk.return_value = iter([[None, ['shoesies', 'shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoes'
        self.naming_scheme.build(self.arti)
        self.arti.set_name.assert_called_once_with('shoes_3')

    @mock.patch('NamingSchemeBuilder.os.walk')
    def test_numerate_name(self, walk):
        walk.return_value = iter([[None, ['shoesies', 'shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoesies'
        self.naming_scheme.build(self.arti)
        self.arti.set_name.assert_called_once_with('shoesies_1')


if __name__ == '__main__':
    unittest.main()
