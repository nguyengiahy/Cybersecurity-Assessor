import unittest
from ..report_generator import generate_report, count_num_positives

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        """
        Set up the default assessment_results and expected report
        """
        self.assessment_results = {
            "c9": {
                "final_result": 1,
                "sentences": [
                    {
                        "text": "Other personal information you provide, including opinions, preferences, goals and previous meditation experience and other personal information contained in product reviews, surveys, or communications.",
                        "raw_label": 0.999651337,
                        "rounded_label": 1,
                    },
                ]
            },
            "c11": {
                "final_result": 0,
                "sentences": [
                    {
                        "text": "Transactional Information: When you make a purchase or return, we collect information about the transaction, such as product description, price, subscription or free trial expiration date, and time and date of the transaction.",
                        "raw_label": 0.000000001,
                        "rounded_label": 0,
                    },
                ]
            },
            "c24": -1,
        }
        self.expected_report = {
            "results": self.assessment_results,
            "total_num_criteria": len(self.assessment_results.keys()),
            "num_criteria_satisfied": 1,
            "percentage_criteria_satisfied": "33.33%",
            "num_pipeline_failed": 1,
            "percentage_failed_pipeline": "33.33%",
            "failed_pipelines": ["c24"]
        }
    
    def test_count_num_positives(self):
        """
        Should count the correct number of positive value
        """
        positives_count = count_num_positives(list(self.assessment_results.values()))
        self.assertEqual(positives_count, 1)

    def test_generate_report(self):
        """
        Should generate report in the expected format
        """
        actual_report = generate_report(self.assessment_results)
        self.assertDictEqual(actual_report, self.expected_report)

if __name__ == "__main__":
    unittest.main()