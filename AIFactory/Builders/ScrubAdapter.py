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
