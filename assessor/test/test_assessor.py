import unittest
from ..assessor import assess_against_threshold, get_sentences_with_labels

class TestAssessor(unittest.TestCase):
    def setUp(self):
        """
        Set up threshold and assessment results and the expected returned dictionary
        """
        self.threshold = 2
        self.raw_assessment_results = [
            [0.124215551],
            [0.000000001],
            [0.457373838],
            [0.759955788],
            [0.531516446],
            [0.864266355],
            [0.999651337],
        ]
        self.raw_assessment_results_without_nested_list = [
            0.124215551,
            0.000000001,
            0.457373838,
            0.759955788,
            0.531516446,
            0.864266355,
            0.999651337,
        ]
        self.rounded_assessment_results = [
            0, 0, 0, 1, 1, 1, 1,
        ]
        self.sentences = [
            "Usage Information: Whenever you use our Services, we collect usage information, such as the sessions you use, videos you view or music you listen to, what screens or features you access, and other similar types of usage information.",
            "Transactional Information: When you make a purchase or return, we collect information about the transaction, such as product description, price, subscription or free trial expiration date, and time and date of the transaction.",
            "Log Information: We collect standard log files when you use our Services, which include the type of web browser you use, app version, access times and dates, pages viewed, your IP address, and the page you visited before navigating to our websites.",
            "Internet or other electronic network activity, such as browsing behavior and information about your usage and interactions with our Services.",
            "Audio, electronic, visual, or similar information, such as profile photo or personal information you may provide during customer support calls and call recordings.",
            "Professional, employment, and education information, such as information we collect about teachers and other administrators in connection with Calm Schools Initiative or CalmHealth Initiative.",
            "Other personal information you provide, including opinions, preferences, goals and previous meditation experience and other personal information contained in product reviews, surveys, or communications.",
        ]
        self.sents_with_labels = [
            {
                "text": "Usage Information: Whenever you use our Services, we collect usage information, such as the sessions you use, videos you view or music you listen to, what screens or features you access, and other similar types of usage information.",
                "raw_label": 0.124215551,
                "rounded_label": 0,
            },
            {
                "text": "Transactional Information: When you make a purchase or return, we collect information about the transaction, such as product description, price, subscription or free trial expiration date, and time and date of the transaction.",
                "raw_label": 0.000000001,
                "rounded_label": 0,
            },
            {
                "text": "Log Information: We collect standard log files when you use our Services, which include the type of web browser you use, app version, access times and dates, pages viewed, your IP address, and the page you visited before navigating to our websites.",
                "raw_label": 0.457373838,
                "rounded_label": 0,
            },
            {
                "text": "Internet or other electronic network activity, such as browsing behavior and information about your usage and interactions with our Services.",
                "raw_label": 0.759955788,
                "rounded_label": 1,
            },
            {
                "text": "Audio, electronic, visual, or similar information, such as profile photo or personal information you may provide during customer support calls and call recordings.",
                "raw_label": 0.531516446,
                "rounded_label": 1,
            },
            {
                "text": "Professional, employment, and education information, such as information we collect about teachers and other administrators in connection with Calm Schools Initiative or CalmHealth Initiative.",
                "raw_label": 0.864266355,
                "rounded_label": 1,
            },
            {
                "text": "Other personal information you provide, including opinions, preferences, goals and previous meditation experience and other personal information contained in product reviews, surveys, or communications.",
                "raw_label": 0.999651337,
                "rounded_label": 1,
            },
        ]
        self.expected_result = {
            "final_result": 1,
            "sentences": self.sents_with_labels,
        }

    def test_get_sentences_with_labels(self):
        """
        Should return sentences with labels in the correct format
        """
        sents = get_sentences_with_labels(self.sentences, self.raw_assessment_results_without_nested_list, self.rounded_assessment_results)
        self.assertListEqual(sents, self.sents_with_labels)

    def test_assess_against_threshold(self):
        """
        Should return final assessment in the correct format
        """
        assessment = assess_against_threshold(self.raw_assessment_results, self.sentences, self.threshold)
        self.assertDictEqual(assessment, self.expected_result)

if __name__ == "__main__":
    unittest.main()