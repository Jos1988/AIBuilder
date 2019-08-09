import os
import warnings

from typing import List

from AIBuilder.AI import AbstractAI


class ModelNameDeterminator:
    """ Generates the name of the Machine Learning model to a unique name by enumerating them.
        This is done because Tensorflow uses the name to create a tensor_board dir,
        and we want each model to be saved in he unique tensor_board dir.

        Names are enumerated based on the dirs in the tensor_board dir, if 5 dirs are found the current model name
        is set as the project name postfixed with "_6". Assuming the dirs where also postfixed the same way.

        If a name has already been assigned it will be used, or enumerated if not unique. This may change in the
        future to allow for continues operations on the same model.

    """

    def __init__(self):
        self.versions = []
        self.existing_names = []

    def determine(self, ml_model: AbstractAI) -> AbstractAI:
        """ Assign name to model, if required. """
        self.existing_names = self._get_logged_names(ml_model)

        if ml_model.get_name() is None or ml_model.get_name() is ml_model.get_project_name():
            new_name = self._generate_name(ml_model)
            ml_model.set_name(new_name)

            return ml_model

        if ml_model.get_name() in self.existing_names:
            ml_model.set_name(ml_model.get_name() + '_1')
            return ml_model

        if ml_model.get_name() is not None:
            return ml_model

        raise RuntimeError(f'{__class__} failed to set name.')

    def _generate_name(self, ml_model: AbstractAI) -> str:
        """ Generates name based on existing dirs in tensor_board dir. """
        for name in self.existing_names:
            version = self._get_version(name=name)
            self.versions.append(version)

        last_version = 0
        if len(self.versions) > 0:
            last_version = max(self.versions)

        new_version = last_version + 1
        new_name = ml_model.get_project_name() + '_' + str(new_version)

        assert new_name not in self.existing_names, f'New model name not unique, {new_name}' \
                                                    f' already in tensor_board folder.'

        return new_name

    @staticmethod
    def _get_logged_names(ml_model: AbstractAI) -> List[str]:
        """ Gets the dir names from the tensor_board dir. """
        tensor_board_path = ml_model.get_log_dir() + '/tensor_board'

        if not os.path.isdir(tensor_board_path):
            warnings.warn(f'Creating missing tensor board dir {tensor_board_path}.')
            os.makedirs(tensor_board_path)

        return next(os.walk(tensor_board_path))[1]

    @staticmethod
    def _get_version(name: str) -> int:
        """ Gets the version of a dir name. """
        exploded = name.split('_')

        version_nr = exploded[-1:][0]
        if version_nr.isnumeric():
            return int(version_nr)

        raise RuntimeError(f'could not resolve version of model name: "{name}".')
