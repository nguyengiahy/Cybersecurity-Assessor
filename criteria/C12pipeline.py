import os
import time
import tensorflow as tf
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Event
from itertools import chain
from ..preprocessors.paragraphs_extractor import ParagraphsExtractor
from ..assessor.assessor import assess_against_threshold
from ..parameters import C12_KEYWORDS, NUM_TOKENS_LOWER_LIMIT
from ..models.bert.utils import init_default_tokeniser

CRITERIA_PATH = "models/bert/saved_models/c12"
CURRENT_DIR = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PARENT_PATH, CRITERIA_PATH)
THRESHOLD = 1  # each pipeline will have its own threshold
NUM_TOP_PARAGRAPHS = 5


class C12pipeline(Process):
    def __init__(self, input_queue: Queue, wait_event: Event, result_queue: Queue):
        """
        Initialise the extractor and load the pre-trained model for C12
        """
        Process.__init__(self)
        self.input_queue = input_queue
        self.wait_event = wait_event
        self.result_queue = result_queue

    def run(self):
        print("C12 Worker process: Starting")
        self.extractor = ParagraphsExtractor()
        self.keywords = C12_KEYWORDS
        init_default_tokeniser()
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print("Exception when loading C12 model: {}".format(e))
            self.model = None
        
        print("C12 Worker process: Ready")
        while(True):
            input = self.input_queue.get()
            self.wait_event.wait()
            assessment_start = time.time()
            self.result_queue.put(self.run_assessment(input))
            print(f"C12 Process assessment time: { time.time() - assessment_start }")

    def get_keywords(self):
        return C12_KEYWORDS

    def get_relevant_score(self, paragraph_stat):
        """Calculate the relevant score of a paragraph based on its statistics

        Args:
            paragraph_stat (dict): a dictionary of different statistics of a paragraph

        Returns:
            (num): the relevant score of a paragraph
        """

        return self.extractor.get_relevant_score_by_keyword_occurence(
            paragraph_stat.get("topic_keyword_occurence"),
            paragraph_stat.get("non_topic_keyword_occurence"),
            weights=[2, 1],
            keywords=self.keywords,
        )

    def run_assessment(self, input_data):
        """Return assessment result for the current policy against C12

        Args:
            input_data (dict): info about the app, containing policy and metadata

        Returns:
            dict or -1: return a dictionary containing the assessment result, or -1 means model fails to load
        """
        if self.model is None:
            return { "c12": -1 }

        if input_data.get("privacyPolicyText") == "":
            # no sentence found inside an empty privacy policy text
            return { "c12": assess_against_threshold([], [], THRESHOLD) }

        paragraphs_stats = input_data.get("statistics", [])
        if len(paragraphs_stats) == 0:
            # no paragraph statistics can be found, i.e. sentences
            return { "c12": assess_against_threshold([], [], THRESHOLD) }

        stats_with_relevant_score = [
            {
                "relevant_score": self.get_relevant_score(stat),
                "sentences": stat.get("sentences"),
            }
            for stat in paragraphs_stats
        ]

        # select the top most relevant paragraphs
        selected_paragraphs_stats = self.extractor.rank_paragraphs_by_relevant_score(
            stats_with_relevant_score, NUM_TOP_PARAGRAPHS
        )
        sentences = list(
            chain.from_iterable(
                stat.get("sentences") for stat in selected_paragraphs_stats
            )
        )
        
        filtered_sents = [
            s
            for s in sentences
            if len(self.extractor.tokenize_text(s)) >= NUM_TOKENS_LOWER_LIMIT
        ]

        # transform sentences into the BERT's expected input format
        processed_sents = self.extractor.process_inputs_for_bert(filtered_sents)

        # asssess the sentences by the machine learning model
        assessments = self.model.predict(processed_sents) if len(processed_sents) > 0 else []

        return { "c12": assess_against_threshold(assessments, filtered_sents, THRESHOLD) }