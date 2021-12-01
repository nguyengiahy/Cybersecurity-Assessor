import unittest
import os

from tensorflow.python.data.ops.dataset_ops import BatchDataset
from ..paragraphs_extractor import ParagraphsExtractor
from ..utils.json_helper import from_json_file

CURRENT_DIR = os.path.dirname(__file__)
TEST_POLICY_PATH = os.path.join(CURRENT_DIR, "fitbit.json")
TEST_SENTENCES_PATH = os.path.join(CURRENT_DIR, "calm_sentences.json")
TEST_PARAGRAPHS_PATH = os.path.join(CURRENT_DIR, "calm_paragraphs.json")
TEST_ZERO_RELEVANCE_PARAGRAPHS_PATH = os.path.join(CURRENT_DIR, "calm_paragraphs_zero_relevance.json")


class TestParagraphsExtractor(unittest.TestCase):
    def setUp(self):
        """
        Set up the instance to be tested before EACH test case
        """
        self.header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        self.privacy_policy = from_json_file(TEST_POLICY_PATH)["privacyPolicyText"]
        self.sentences = from_json_file(TEST_SENTENCES_PATH)
        self.paragraphs = from_json_file(TEST_PARAGRAPHS_PATH)
        self.zero_paragraphs = from_json_file(TEST_ZERO_RELEVANCE_PARAGRAPHS_PATH)
        self.extractor = ParagraphsExtractor()

    def test_get_segments_by_html_tags(self):
        """
        Segment header tag should be either h1, h2, h3, h4, h5, h6
        Segment header content should be string
        """
        actual_segments = self.extractor.get_segments_by_html_tags(self.privacy_policy)

        for segment in actual_segments:
            self.assertIn(segment["header"]["tag"], self.header_tags)
            self.assertIsInstance(segment["header"]["content"], str)

    def test_get_paragraphs_per_segment(self):
        """
        Paragraph header tag should not be either h1, h2, h3, h4, h5, h6
        Paragraph content should be string
        """
        actual_segments = self.extractor.get_segments_by_html_tags(self.privacy_policy)

        for segment in actual_segments:
            for para in segment["paragraphs"]:
                self.assertNotIn(para["tag"], self.header_tags)
                self.assertIsInstance(para["content"], str)

    def test_get_sentences_from_single_paragraph(self):
        """
        get_sentences_from_single_paragraph must be able to tokenize sentences with tolerance to the existence of abbreviation noise,
        """
        # test simple paragraph. No noise.
        p1 = "Accessing and Exporting Data. By logging into your account, you can access much of your personal information, including your dashboard with your daily exercise and activity statistics. Using your account settings, you can also download information in a commonly used file format, including data about your activities, body, foods, and sleep. Learn more here."
        expected_sents1 = [
            "Accessing and Exporting Data.",
            "By logging into your account, you can access much of your personal information, including your dashboard with your daily exercise and activity statistics.",
            "Using your account settings, you can also download information in a commonly used file format, including data about your activities, body, foods, and sleep.",
            "Learn more here.",
        ]
        actual_sents1 = self.extractor.get_sentences_from_single_paragraph(p1)
        self.assertIsInstance(
            actual_sents1,
            list,
            f"TypeError: expected list, found {type(actual_sents1)}.",
        )
        self.assertEqual(
            len(actual_sents1),
            len(expected_sents1),
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents1,
            expected_sents1,
            "ValueError: actual list is not the same as expected list.",
        )

        # test empty paragraph
        p2 = ""
        expected_sents2 = []
        actual_sents2 = self.extractor.get_sentences_from_single_paragraph(p2)
        self.assertIsInstance(
            actual_sents2,
            list,
            f"TypeError: expected list, found {type(actual_sents2)}.",
        )
        self.assertEqual(
            len(actual_sents2),
            0,
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents2,
            expected_sents2,
            "ValueError: actual list is not the same as expected list.",
        )

        # test paragraph with 'no.'
        p3 = "E.g. Jersey no. 5 is the best. But I don't like no. 6."
        expected_sents3 = ["E.g. Jersey no. 5 is the best.", "But I don't like no. 6."]
        actual_sents3 = self.extractor.get_sentences_from_single_paragraph(p3)

        self.assertIsInstance(
            actual_sents3,
            list,
            f"TypeError: expected list, found {type(actual_sents3)}.",
        )
        self.assertEqual(
            len(actual_sents3),
            len(expected_sents3),
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents3,
            expected_sents3,
            "ValueError: actual list is not the same as expected list.",
        )

        # test paragraph with more noise ("i.e.", "C.V.V.")
        p4 = "Some Fitbit devices support payments and transactions with third parties. If you activate this feature, you must provide certain information for identification and verification, i.e. your name, credit, debit or other card number, card expiration date, and C.V.V code. This information is encrypted and sent to your card network, which upon approval sends back to your device a token, which is a set of random digits for engaging in transactions without exposing your card number."
        expected_sents4 = [
            "Some Fitbit devices support payments and transactions with third parties.",
            "If you activate this feature, you must provide certain information for identification and verification, i.e. your name, credit, debit or other card number, card expiration date, and C.V.V code.",
            "This information is encrypted and sent to your card network, which upon approval sends back to your device a token, which is a set of random digits for engaging in transactions without exposing your card number.",
        ]

        actual_sents4 = self.extractor.get_sentences_from_single_paragraph(p4)
        self.assertIsInstance(
            actual_sents4,
            list,
            f"TypeError: expected list, found {type(actual_sents4)}.",
        )
        self.assertEqual(
            len(actual_sents4),
            len(expected_sents4),
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents4,
            expected_sents4,
            "ValueError: actual list is not the same as expected list.",
        )

        # test with numeric and version noise: v1.0, v5.4, 2.0
        # test with abbreviation noise: e.g.
        p5 = "The policy below, as well as the Google documents, e.g. Google APIs Terms of Service, Google API Services User Data Policy v1.0, Google Fit Developer Terms and Conditions v5.4, Google Fit Developer Guide, and OAuth 2.0 Policies govern the use of and access to Google Fit APIs and associated Google Fit user data. You must also comply with all applicable laws and regulations. Additional policies made available by Google from time to time may apply. In the event of a conflict between this policy or any other terms with regard to accessing user data, this Developer and User Data Policy controls."

        expected_sents5 = [
            "The policy below, as well as the Google documents, e.g. Google APIs Terms of Service, Google API Services User Data Policy v1.0, Google Fit Developer Terms and Conditions v5.4, Google Fit Developer Guide, and OAuth 2.0 Policies govern the use of and access to Google Fit APIs and associated Google Fit user data.",
            "You must also comply with all applicable laws and regulations.",
            "Additional policies made available by Google from time to time may apply.",
            "In the event of a conflict between this policy or any other terms with regard to accessing user data, this Developer and User Data Policy controls.",
        ]
        actual_sents5 = self.extractor.get_sentences_from_single_paragraph(p5)
        self.assertIsInstance(
            actual_sents5,
            list,
            f"TypeError: expected list, found {type(actual_sents5)}.",
        )
        self.assertEqual(
            len(actual_sents5),
            len(expected_sents5),
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents5,
            expected_sents5,
            "ValueError: actual list is not the same as expected list.",
        )

        # test with email noise
        p6 = "Contact support@customer-service.samsung.kr in case you found a minor user using our service. We will try our best to protect children."
        expected_sents6 = [
            "Contact support@customer-service.samsung.kr in case you found a minor user using our service.",
            "We will try our best to protect children.",
        ]
        actual_sents6 = self.extractor.get_sentences_from_single_paragraph(p6)
        self.assertIsInstance(
            actual_sents6,
            list,
            f"TypeError: expected list, found {type(actual_sents6)}.",
        )
        self.assertEqual(
            len(actual_sents6),
            len(expected_sents6),
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents6,
            expected_sents6,
            "ValueError: actual list is not the same as expected list.",
        )

        # test with etc
        p7 = "For example, an excerpt from Pinterest's Privacy Policy agreement clearly describes the information Pinterest collects from its users as well as from any other source that users enable Pinterest to gather information from etc. The information that the user voluntarily gives includes names, photos, pins, likes, email address, and/or phone number etc., all of which is regarded as personal information."
        expected_sents7 = [
            "For example, an excerpt from Pinterest's Privacy Policy agreement clearly describes the information Pinterest collects from its users as well as from any other source that users enable Pinterest to gather information from etc.",
            "The information that the user voluntarily gives includes names, photos, pins, likes, email address, and/or phone number etc., all of which is regarded as personal information.",
        ]
        actual_sents7 = self.extractor.get_sentences_from_single_paragraph(p7)
        self.assertIsInstance(
            actual_sents7,
            list,
            f"TypeError: expected list, found {type(actual_sents7)}.",
        )
        self.assertEqual(
            len(actual_sents7),
            len(expected_sents7),
            "ValueError: expected result and actual result do not have the same length.",
        )
        self.assertListEqual(
            actual_sents7,
            expected_sents7,
            "ValueError: actual list is not the same as expected list.",
        )

    def test_get_topic_sentence(self):
        p1 = "Accessing and Exporting Data. By logging into your account, you can access much of your personal information, including your dashboard with your daily exercise and activity statistics. Using your account settings, you can also download information in a commonly used file format, including data about your activities, body, foods, and sleep. Learn more here."
        actual_topic1 = self.extractor.get_topic_sentence(p1)
        self.assertEqual(
            actual_topic1,
            "Accessing and Exporting Data.",
            f"ValueError: incorrect result, found {actual_topic1}",
        )

        p2 = ""
        actual_topic2 = self.extractor.get_topic_sentence(p2)
        self.assertEqual(
            actual_topic2, "", f"ValueError: incorrect result, found {actual_topic2}"
        )

        p3 = "Jersey no. 5 is the best. But I don't like no. 6."
        actual_topic3 = self.extractor.get_topic_sentence(p3)
        self.assertEqual(
            actual_topic3,
            "Jersey no. 5 is the best.",
            f"ValueError: incorrect result, found {actual_topic3}",
        )

        p4 = "If you activate this feature, you must provide certain information for identification and verification, i.e. your name, credit, debit or other card number, card expiration date, and C.V.V code. This information is encrypted and sent to your card network, which upon approval sends back to your device a token, which is a set of random digits for engaging in transactions without exposing your card number."
        actual_topic4 = self.extractor.get_topic_sentence(p4)
        self.assertEqual(
            actual_topic4,
            "If you activate this feature, you must provide certain information for identification and verification, i.e. your name, credit, debit or other card number, card expiration date, and C.V.V code.",
            f"ValueError: incorrect result, found {actual_topic4}",
        )

        p5 = "The policy below, as well as the Google documents, e.g. Google APIs Terms of Service, Google API Services User Data Policy v1.0, Google Fit Developer Terms and Conditions v5.4, Google Fit Developer Guide, and OAuth 2.0 Policies govern the use of and access to Google Fit APIs and associated Google Fit user data. You must also comply with all applicable laws and regulations. Additional policies made available by Google from time to time may apply. In the event of a conflict between this policy or any other terms with regard to accessing user data, this Developer and User Data Policy controls."
        actual_topic5 = self.extractor.get_topic_sentence(p5)
        self.assertEqual(
            actual_topic5,
            "The policy below, as well as the Google documents, e.g. Google APIs Terms of Service, Google API Services User Data Policy v1.0, Google Fit Developer Terms and Conditions v5.4, Google Fit Developer Guide, and OAuth 2.0 Policies govern the use of and access to Google Fit APIs and associated Google Fit user data.",
            f"ValueError: incorrect result, found {actual_topic5}",
        )

        p6 = "Health Kit - You will also have an option to permit us to import health data into the App from third-party services such as Google Fit."
        actual_topic6 = self.extractor.get_topic_sentence(p6)
        self.assertEqual(
            actual_topic6,
            "Health Kit - You will also have an option to permit us to import health data into the App from third-party services such as Google Fit.",
            f"ValueError: incorrect result, found {actual_topic6}",
        )

    def test_tokenize_word(self):
        p1 = """Google Analytics. We use Google Analytics to track website traffic."""

        actual_tokens = self.extractor.tokenize_text(p1, need_lemma=False)
        print(actual_tokens)
        expected_tokens = [
            "Google",
            "Analytics",
            "We",
            "use",
            "Google",
            "Analytics",
            "to",
            "track",
            "website",
            "traffic",
        ]
        for actual_token, expected_token in zip(actual_tokens, expected_tokens):
            self.assertEqual(actual_token, expected_token, "ValueError: unequal tokens")

    def test_count_keyword(self):
        p1 = """
        Google Analytics.
        We use Google Analytics to track website traffic.
        Google Analytics is a web analytics service offered by Google.
        Google uses the data collected to track and monitor the use of our websites.
        This data is shared with other Google services.
        Google may use the collected data to contextualise and personalise the ads of its own advertising network.
        For more information on the privacy practices of Google, please visit the Google Privacy Terms web page
        """
        occurence1 = self.extractor.count_keyword(p1, "collect", need_lemma=True)
        self.assertEqual(
            occurence1, 2, f"ValueError: incorrect result, found {occurence1}"
        )
        occurence2 = self.extractor.count_keyword(p1, "collect", need_lemma=False)
        self.assertEqual(
            occurence2, 0, f"ValueError: incorrect result, found {occurence1}"
        )

        p2 = """
            Your Personal Information will never be shared without your consent:
            We will not sell, lease, or rent your individual-level protected health information to any third party without your consent.
            We do not share customer data with any public databases.
            We will not provide any Personal Information to an insurance company or employer.
            We will not provide information to law enforcement or regulatory authorities unless required by law to comply with a valid court order, subpoena, or search warrant for genetic or Personal Information.
        """
        occurence3 = self.extractor.count_keyword(p2, "share", need_lemma=True)
        self.assertEqual(
            occurence3, 2, f"ValueError: incorrect result, found {occurence3}"
        )
        occurence4 = self.extractor.count_keyword(p2, "share", need_lemma=False)
        self.assertEqual(
            occurence4, 1, f"ValueError: incorrect result, found {occurence4}"
        )

    def test_count_keywords(self):
        p1 = """
        Google Analytics.
        We use Google Analytics to track website traffic.
        Google Analytics is a web analytics service offered by Google.
        Google uses the data collected to track and monitor the use of our websites.
        This data is shared with other Google services.
        Google may use the collected data to contextualise and personalise the ads of its own advertising network.
        For more information on the privacy practices of Google, please visit the Google Privacy Terms web page
        """
        occurence1 = self.extractor.count_keywords(
            p1, ["collect", "Google"], need_lemma=True
        )
        self.assertDictEqual(
            occurence1,
            {"collect": 2, "google": 9},
            f"ValueError: incorrect result, found {occurence1}",
        )

    def test_get_paragraph_statistics(self):
        p1 = """
            Your Personal Information will never be shared without your consent.
            We will not sell, lease, or rent your individual-level protected health information to any third party without your consent.
            We do not share customer data with any public databases.
            We will not provide any Personal Information to an insurance company or employer.
            We will not provide information to law enforcement or regulatory authorities unless required by law to comply with a valid court order, subpoena, or search warrant for genetic or Personal Information.
        """

        expected = self.extractor.get_paragraph_statistics(
            p1, ["share", "consent", "provide"]
        )
        self.assertIsInstance(expected, dict, "Result must be a dictionary")

        self.assertDictEqual(
            expected.get("topic_keyword_occurence"),
            {"share": 1, "consent": 1, "provide": 0},
        )
        self.assertDictEqual(
            expected.get("non_topic_keyword_occurence"),
            {"share": 1, "consent": 1, "provide": 2},
        )

    def test_get_relevant_score_by_keyword_occurence(self):
        """
        Testing relevant score provided by
        ParagraphExtractor.get_relevant_score_by_keyword_occurence()
        """
        actual_score1 = self.extractor.get_relevant_score_by_keyword_occurence(
            {"ad": 1, "provide": 2},
            {"ad": 3, "gather": 2, "provide": 4},
            weights=[1, 1],
            keywords=["ad", "provide"],
        )
        self.assertAlmostEqual(
            actual_score1,
            5,
            places=3,
            msg="ValueError: Calculated score does not match",
        )

        actual_score2 = self.extractor.get_relevant_score_by_keyword_occurence(
            {"ad": 1, "provide": 2},
            {"ad": 3, "gather": 2, "provide": 4},
            weights=[3, 1],
            keywords=["ad", "provide"],
        )
        self.assertAlmostEqual(
            actual_score2,
            4,
            places=3,
            msg="ValueError: Calculated score does not match",
        )

        actual_score3 = self.extractor.get_relevant_score_by_keyword_occurence(
            {"collect": 3, "collection": 1, "information": 2},
            {"collect": 2, "collection": 3, "information": 5},
            weights=[],
            keywords=["collect", "information"],
        )
        self.assertAlmostEqual(
            actual_score3,
            6,
            places=3,
            msg="ValueError: Calculated score does not match",
        )

    def test_get_paragraphs__yogaquote(self):
        policy_html = """
            <body>
                <h1>Privacy Policy of Daily Yoga Quotes - Inspirational Yoga Quote of the Day</h1>
                <p>This Application collects some Personal Data from its Users.</p>
                <br/>
                <p>This document can be printed for reference by using the print command in the settings of any browser.</p>
                <h1>Privacy Policy of Daily Yoga Quotes - Inspirational Yoga Quote of the Day</h1>
                <p>This Application collects some Personal Data from its Users.</p>
                <br/>
                <p>This document can be printed for reference by using the print command in the settings of any browser.</p>
                <h1>Privacy Policy of Daily Yoga Quotes - Inspirational Yoga Quote of the Day</h1>
                <p>This Application collects some Personal Data from its Users.</p>
                <br/>
                <p>This document can be printed for reference by using the print command in the settings of any browser.</p>
                <h1>Privacy Policy of Daily Yoga Quotes - Inspirational Yoga Quote of the Day</h1>
                <p>This Application collects some Personal Data from its Users.</p>
                <br/>
                <p>This document can be printed for reference by using the print command in the settings of any browser.</p>
                <h1>Privacy Policy of Daily Yoga Quotes - Inspirational Yoga Quote of the Day</h1>
                <p>This Application collects some Personal Data from its Users.</p>
                <br/>
                <p>This document can be printed for reference by using the print command in the settings of any browser.</p>
                <h1>Privacy Policy of Daily Yoga Quotes - Inspirational Yoga Quote of the Day</h1>
                <p>This Application collects some Personal Data from its Users.</p>
                <br/>
                <p>This document can be printed for reference by using the print command in the settings of any browser.</p>
            <body>
        """
        actual = self.extractor.get_paragraphs(policy_html)
        self.assertIsInstance(actual, list, msg="Invalid type, expected list")
        self.assertEqual(len(actual), 12, msg="Invalid length")
        self.assertEqual(
            actual[0],
            "This Application collects some Personal Data from its Users.",
            msg="Incorrect value",
        )
        self.assertEqual(
            actual[1],
            "This document can be printed for reference by using the print command in the settings of any browser.",
            msg="Incorrect value",
        )

    def test_get_paragraphs__myfitnesspal(self):
        policy_html = """
            <body>
                <h2>Information About Your Personal Data</h2>
                <p>This Privacy Policy relates to data about you, your devices, and your interaction with our Services.</p>
                <p><strong>Personal Data</strong> is information that can be used to identify you, directly or indirectly, 
                alone or together with other information. This includes things such as your full name, email address, phone 
                number, device IDs, certain cookie and network identifiers, and “Fitness and Wellness Data.”
                </p>
                </h3>When you register for an account or interact with our Services</h2>
                <p>We collect Personal Data when you use or interact with our Services, including when you register for a 
                MyFitnessPal account, purchase a Premium Subscription (including processing of payment), or otherwise use the 
                Services (e.g., browse the content available on the Services), and when you ask us to customize our Services. 
                This Personal Data may include name, photo, username and password, email address, date of birth, gender, 
                payment information and general location data.
                </p>
                <h2>Information About Your Personal Data</h2>
                <p>This Privacy Policy relates to data about you, your devices, and your interaction with our Services.</p>
                <p><strong>Personal Data</strong> is information that can be used to identify you, directly or indirectly, 
                alone or together with other information. This includes things such as your full name, email address, phone 
                number, device IDs, certain cookie and network identifiers, and “Fitness and Wellness Data.”
                </p>
                </h3>When you register for an account or interact with our Services</h2>
                <p>We collect Personal Data when you use or interact with our Services, including when you register for a 
                MyFitnessPal account, purchase a Premium Subscription (including processing of payment), or otherwise use the 
                Services (e.g., browse the content available on the Services), and when you ask us to customize our Services. 
                This Personal Data may include name, photo, username and password, email address, date of birth, gender, 
                payment information and general location data.
                </p>
                <h2>Information About Your Personal Data</h2>
                <p>This Privacy Policy relates to data about you, your devices, and your interaction with our Services.</p>
                <p><strong>Personal Data</strong> is information that can be used to identify you, directly or indirectly, 
                alone or together with other information. This includes things such as your full name, email address, phone 
                number, device IDs, certain cookie and network identifiers, and “Fitness and Wellness Data.”
                </p>
                </h3>When you register for an account or interact with our Services</h2>
                <p>We collect Personal Data when you use or interact with our Services, including when you register for a 
                MyFitnessPal account, purchase a Premium Subscription (including processing of payment), or otherwise use the 
                Services (e.g., browse the content available on the Services), and when you ask us to customize our Services. 
                This Personal Data may include name, photo, username and password, email address, date of birth, gender, 
                payment information and general location data.
                </p>
                <h2>Information About Your Personal Data</h2>
                <p>This Privacy Policy relates to data about you, your devices, and your interaction with our Services.</p>
                <p><strong>Personal Data</strong> is information that can be used to identify you, directly or indirectly, 
                alone or together with other information. This includes things such as your full name, email address, phone 
                number, device IDs, certain cookie and network identifiers, and “Fitness and Wellness Data.”
                </p>
                </h3>When you register for an account or interact with our Services</h2>
                <p>We collect Personal Data when you use or interact with our Services, including when you register for a 
                MyFitnessPal account, purchase a Premium Subscription (including processing of payment), or otherwise use the 
                Services (e.g., browse the content available on the Services), and when you ask us to customize our Services. 
                This Personal Data may include name, photo, username and password, email address, date of birth, gender, 
                payment information and general location data.
                </p>
                <h2>Information About Your Personal Data</h2>
                <p>This Privacy Policy relates to data about you, your devices, and your interaction with our Services.</p>
                <p><strong>Personal Data</strong> is information that can be used to identify you, directly or indirectly, 
                alone or together with other information. This includes things such as your full name, email address, phone 
                number, device IDs, certain cookie and network identifiers, and “Fitness and Wellness Data.”
                </p>
                </h3>When you register for an account or interact with our Services</h2>
                <p>We collect Personal Data when you use or interact with our Services, including when you register for a 
                MyFitnessPal account, purchase a Premium Subscription (including processing of payment), or otherwise use the 
                Services (e.g., browse the content available on the Services), and when you ask us to customize our Services. 
                This Personal Data may include name, photo, username and password, email address, date of birth, gender, 
                payment information and general location data.
                </p>
            <body>
        """
        actual = self.extractor.get_paragraphs(policy_html)
        self.assertIsInstance(actual, list, msg="Invalid type, expected list")
        self.assertEqual(len(actual), 15, msg="Invalid length")

        self.assertEqual(
            actual[0],
            "This Privacy Policy relates to data about you, your devices, and your interaction with our Services.",
            msg="Incorrect value",
        )
        # observe behavior
        # Spacy lemmatization
        # Data -> data
        # data -> datum

    def test_process_inputs_for_bert(self):
        """
        The structure of BatchDataset is quite complex, so we'll just test if we return the correct type or not
        """
        actual_inputs = self.extractor.process_inputs_for_bert(self.sentences)
        self.assertIsInstance(actual_inputs, BatchDataset)

    def test_rank_paragraphs_with_same_relevance(self):
        expected_paragraphs = self.zero_paragraphs

        actual_paragraphs = self.extractor.rank_paragraphs_by_relevant_score(self.zero_paragraphs, 5)

        self.assertListEqual(expected_paragraphs, actual_paragraphs)


if __name__ == "__main__":
    unittest.main()
