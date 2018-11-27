import unittest
from unittest import mock
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
from AIBuilder.AIFactory.Specifications.BasicSpecifications import Descriptor
import AIBuilder.DataScrubbing as scrubber


class ScrubAdapter(Builder):

    def __init__(self, scrubbers: list = None):
        self.and_scrubber = scrubber.AndScrubber()
        self.descriptor = Descriptor('scrubbers', None)
        if scrubbers is not None:
            for new_scrubber in scrubbers:
                assert isinstance(new_scrubber, scrubber.Scrubber)
                self.add_scrubber(new_scrubber)

    @property
    def dependent_on(self) -> list:
        return [self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.SCRUBBER

    def add_scrubber(self, scrubber: scrubber.Scrubber):
        self.and_scrubber.add_scrubber(scrubber)
        self.descriptor.add_description(scrubber.__class__.__name__)

    def validate(self):
        pass

    def build(self, neural_net: AbstractAI):
        training_data = neural_net.training_data
        validation_data = neural_net.evaluation_data

        self.and_scrubber.validate_metadata(training_data.metadata)
        self.and_scrubber.scrub(training_data)

        self.and_scrubber.validate_metadata(validation_data.metadata)
        self.and_scrubber.scrub(validation_data)


class TestScrubAdapter(unittest.TestCase):

    def setUp(self):
        self.scrubber_three = mock.Mock('ScrubAdapter.scrubber')
        self.scrubber_four = mock.Mock('ScrubAdapter.scrubber')
        self.scrubber_one = mock.Mock(spec=scrubber.Scrubber)
        self.scrubber_two = mock.Mock(spec=scrubber.Scrubber)
        self.scrub_adapter = ScrubAdapter([self.scrubber_one, self.scrubber_two])
        self.scrub_adapter.add_scrubber(self.scrubber_three)
        self.scrub_adapter.add_scrubber(self.scrubber_four)

    def test_add_scrubber(self):
        self.assertIn(self.scrubber_one, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_two, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_three, self.scrub_adapter.and_scrubber.scrubber_list)
        self.assertIn(self.scrubber_four, self.scrub_adapter.and_scrubber.scrubber_list)

    def test_build(self):
        arti = mock.Mock('ScrubAdapter.AbstractAI')
        training_data = mock.Mock('ScrubAdapter.Data.DataModel')
        training_data.metadata = mock.Mock(name='training_metadata')

        evaluation_data = mock.patch('ScrubAdapter.Data.DataModel')
        evaluation_data.metadata = mock.Mock(name='validation_metadata')

        and_scrubber = mock.Mock(name='and_scrubber')
        and_scrubber.validate_metadata = mock.Mock(name='and_scrubber_validate_metadata')
        and_scrubber.scrub = mock.Mock(name='and_scrubber_scrub')

        arti.training_data = training_data
        arti.evaluation_data = evaluation_data
        self.scrub_adapter.and_scrubber = and_scrubber

        self.scrub_adapter.build(arti)

        and_scrubber.validate_metadata.assert_any_call(training_data.metadata),
        and_scrubber.scrub.assert_any_call(training_data),
        and_scrubber.validate_metadata.assert_any_call(evaluation_data.metadata),
        and_scrubber.scrub.assert_any_call(evaluation_data)


if __name__ == '__main__':
    unittest.main()
