import unittest
from unittest import mock

from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
import AIBuilder.DataScrubbing as scrubber


class ScrubAdapter(Builder):

    def __init__(self):
        self.and_scrubber = scrubber.AndScrubber()

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL]

    @property
    def ingredient_type(self) -> str:
        return self.SCRUBBER

    def add_scrubber(self, scrubber: scrubber.Scrubber):
        self.and_scrubber.add_scrubber(scrubber)

    def validate(self):
        pass

    def build(self, neural_net: AbstractAI):
        training_data = neural_net.training_data
        validation_data = neural_net.validation_data

        self.and_scrubber.validate_metadata(training_data.metadata)
        self.and_scrubber.scrub(training_data)

        self.and_scrubber.validate_metadata(validation_data.metadata)
        self.and_scrubber.scrub(validation_data)


class TestScrubAdapter(unittest.TestCase):

    def setUp(self):
        self.scrubber_one = mock.patch('AIFactory.scrubber')
        self.scrubber_two = mock.patch('AIFactory.scrubber')

        self.scrub_adapter = ScrubAdapter()
        self.scrub_adapter.add_scrubber(self.scrubber_one)
        self.scrub_adapter.add_scrubber(self.scrubber_two)

    def test_add_scrubber(self):
        self.assertIn(self.scrubber_one, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_two, self.scrub_adapter.and_scrubber.scrubber_list)

    @mock.patch('AIFactory.AI')
    def test_build(self, mock_ai):
        training_data = mock.patch('AIFactory.Data.DataModel')
        training_data.metadata = mock.Mock(name='training_metadata')

        validation_data = mock.patch('AIFactory.Data.DataModel')
        validation_data.metadata = mock.Mock(name='validation_metadata')

        and_scrubber = mock.Mock(name='and_scrubber')
        and_scrubber.validate_metadata = mock.Mock(name='and_scrubber_validate_metadata')
        and_scrubber.scrub = mock.Mock(name='and_scrubber_scrub')

        mock_ai.training_data = training_data
        mock_ai.validation_data = validation_data
        self.scrub_adapter.and_scrubber = and_scrubber

        self.scrub_adapter.build(mock_ai)

        and_scrubber.validate_metadata.assert_any_call(training_data.metadata),
        and_scrubber.scrub.assert_any_call(training_data),
        and_scrubber.validate_metadata.assert_any_call(validation_data.metadata),
        and_scrubber.scrub.assert_any_call(validation_data)
