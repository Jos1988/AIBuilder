import unittest
from unittest import mock

from AIBuilder.AIFactory.Logistics import ModelNameDeterminator


class TestNamingScheme(unittest.TestCase):

    def setUp(self):
        self.naming_scheme = ModelNameDeterminator()
        self.arti = mock.Mock('test_Builders.NamingSchemeBuilder.AIBuilder.AI')
        self.arti.get_log_dir = mock.Mock()
        self.arti.get_log_dir.return_value = '../../../builder projects/log'
        self.arti.get_project_name = mock.Mock()
        self.arti.set_name = mock.Mock()
        self.arti.get_name = mock.Mock()
        self.arti.get_name.return_value = None

    @mock.patch('AIBuilder.AIFactory.Logistics.os.walk')
    def test_generate_name(self, walk):
        walk.return_value = iter([[None, ['shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoes'
        self.naming_scheme.determine(self.arti)
        self.arti.set_name.assert_called_once_with('shoes_3')

    @mock.patch('AIBuilder.AIFactory.Logistics.os.walk')
    def test_generate_name_no_version(self, walk):
        walk.return_value = iter([[None, ['shoesies', 'shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoes'
        with self.assertRaises(RuntimeError, msg='could not resolve version of model name "shoesies"'):
            self.naming_scheme.determine(self.arti)

    @mock.patch('AIBuilder.AIFactory.Logistics.os.walk')
    def test_numerate_name(self, walk):
        walk.return_value = iter([[None, ['shoes_1', 'shoes_2']]])
        self.arti.get_project_name.return_value = 'shoes'
        self.naming_scheme.determine(self.arti)
        self.arti.set_name.assert_called_once_with('shoes_3')

    @mock.patch('AIBuilder.AIFactory.Logistics.os.walk')
    def test_extra_underscore(self, walk):
        walk.return_value = iter([[None, ['my_shoes_1', 'my_shoes_2']]])
        self.arti.get_project_name.return_value = 'my_shoes'
        self.naming_scheme.determine(self.arti)
        self.arti.set_name.assert_called_once_with('my_shoes_3')


if __name__ == '__main__':
    unittest.main()
