import multiprocessing
import time
from itertools import chain
from multiprocessing import Queue
from multiprocessing import Event
from .criteria.C9pipeline import C9pipeline
from .criteria.C10pipeline import C10pipeline
from .criteria.C11pipeline import C11pipeline
from .criteria.C12pipeline import C12pipeline
from .criteria.C24pipeline import C24pipeline
from .criteria.C49pipeline import C49pipeline
from .report_generator.report_generator import generate_report
from .preprocessors.paragraphs_extractor import ParagraphsExtractor
from .parameters import INPUT_QUEUE_SIZE, NUM_CRITERIA

class Pipeline:
    def __init__(self):
        self.assessment_queue = Queue(NUM_CRITERIA)
        self.instantiate_criteria_queues()
        self.assessment_begin_event = Event()
        self.instantiate_criteria_pipelines()
        self.criteria_pipelines = {
            "c9": { "pipeline": self.c9pipeline, "input_queue": self.c9input_queue },
            "c10": { "pipeline": self.c10pipeline, "input_queue": self.c10input_queue },
            "c11": { "pipeline": self.c11pipeline, "input_queue": self.c11input_queue },
            "c12": { "pipeline": self.c12pipeline, "input_queue": self.c12input_queue },
            "c24": { "pipeline": self.c24pipeline, "input_queue": self.c24input_queue },
            "c49": { "pipeline": self.c49pipeline, "input_queue": self.c49input_queue },
            # TODO: add more criteria pipelines
        }

        print("Number of processors: " + str(multiprocessing.cpu_count()))

        for pipeline_process in self.get_pipelines():
            pipeline_process.start()

        self.extractor = ParagraphsExtractor()
        self.keywords = list(
            chain.from_iterable(
                pipeline.get_keywords() for pipeline in self.get_pipelines()
            )
        )

    def get_pipelines(self):
        return [value.get("pipeline") for value in self.criteria_pipelines.values()]

    def get_input_queues(self):
        return [value.get("input_queue") for value in self.criteria_pipelines.values()]

    def instantiate_criteria_queues(self):
        self.c9input_queue  = Queue(INPUT_QUEUE_SIZE)
        self.c10input_queue = Queue(INPUT_QUEUE_SIZE)
        self.c11input_queue = Queue(INPUT_QUEUE_SIZE)
        self.c12input_queue = Queue(INPUT_QUEUE_SIZE)
        self.c24input_queue = Queue(INPUT_QUEUE_SIZE)
        self.c49input_queue = Queue(INPUT_QUEUE_SIZE)
        # TODO: add more queues

    def instantiate_criteria_pipelines(self):
        self.c9pipeline  = C9pipeline(self.c9input_queue, self.assessment_begin_event, self.assessment_queue)
        self.c10pipeline = C10pipeline(self.c10input_queue, self.assessment_begin_event, self.assessment_queue)
        self.c11pipeline = C11pipeline(self.c11input_queue, self.assessment_begin_event, self.assessment_queue)
        self.c12pipeline = C12pipeline(self.c12input_queue, self.assessment_begin_event, self.assessment_queue)
        self.c24pipeline = C24pipeline(self.c24input_queue, self.assessment_begin_event, self.assessment_queue)
        self.c49pipeline = C49pipeline(self.c49input_queue, self.assessment_begin_event, self.assessment_queue)
        # TODO: add more pipelines

    def run_assessment(self, input_data, criteria):
        """Start the assessment pipeline & trigger individual pipeline for each criteria. A report is generated at the end to summarise the results

        Args:
            input_data (dict): Info about the app, including policy and metadata
            criteria (str): criteria name (e.g c9, c10, c24). If criteria is None, pipelines of all criteria will be triggered to run.

        Returns:
            (dict): Summary of the results
        """
        # Check that the criteria is correct, if specified, first to skip preprocessing if there is an issue
        if criteria and not criteria in self.criteria_pipelines:
            return { "message": "Criteria does not exist :(" }

        start_assessment_time = time.time()
        privacy_policy_html = input_data.get("privacyPolicyText", "")
        paragraphs = self.extractor.get_paragraphs(privacy_policy_html)

        ###################################################################
        stats_start = time.time()
        # get the paragraphs statistics that are used by all criteria
        paragraphs_statistics = [
            self.extractor.get_paragraph_statistics(paragraph, set(self.keywords))
            for paragraph in paragraphs
        ]
        data = {"statistics": paragraphs_statistics, **input_data}
        stats_end = time.time()
        ###################################################################

        assessment_results = {}
        try:
            ###################################################################
            pipeline_specific_start = time.time()
            if criteria is None:
                for input_queue in self.get_input_queues():
                    input_queue.put(data, block=False)

                self.assessment_begin_event.set()

                assessment_results = {}
                while len(assessment_results) < len(self.criteria_pipelines):
                    single_assessment = self.assessment_queue.get(block=True)
                    assessment_results.update(single_assessment)

                self.assessment_begin_event.clear()

            else:
                self.criteria_pipelines[criteria]["input_queue"].put(data, block=False)

                self.assessment_begin_event.set()

                assessment_results = self.assessment_queue.get(block=True)
                self.assessment_begin_event.clear()

            pipeline_specific_end = time.time()
            ###################################################################
            print(f"\nCalculate paragraphs statistics (topic/keywords): {stats_end - stats_start}s")
            print(f"Assessment processes (sending input to the processes and waiting for responses): {pipeline_specific_end - pipeline_specific_start}s.")
            print(f"Entire pipeline: {time.time() - start_assessment_time}s")
            return generate_report(assessment_results)

        except Exception as e:
            return {
                "message": "Something went wrong :( {}. Please try again later.".format(
                    e
                )
            }