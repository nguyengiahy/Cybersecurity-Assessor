import unittest
import os
import tensorflow as tf
import itertools
from .utils import init_default_tokeniser, to_feature_map

C9_CRITERIA_PATH = 'bert/saved_models/c9'
C10_CRITERIA_PATH = 'bert/saved_models/c10'
C11_CRITERIA_PATH = 'bert/saved_models/c11'
C12_CRITERIA_PATH = 'bert/saved_models/c12'
C24_CRITERIA_PATH = 'bert/saved_models/c24'
C49_CRITERIA_PATH = 'bert/saved_models/c49'

CURRENT_DIR = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CURRENT_DIR)

C9_MODEL_PATH = os.path.join(PARENT_PATH, C9_CRITERIA_PATH)
C10_MODEL_PATH = os.path.join(PARENT_PATH, C10_CRITERIA_PATH)
C11_MODEL_PATH = os.path.join(PARENT_PATH, C11_CRITERIA_PATH)
C12_MODEL_PATH = os.path.join(PARENT_PATH, C12_CRITERIA_PATH)
C24_MODEL_PATH = os.path.join(PARENT_PATH, C24_CRITERIA_PATH)
C49_MODEL_PATH = os.path.join(PARENT_PATH, C49_CRITERIA_PATH)

# Set the threshold low as this test is not about accuracy, but more about documentation
TEST_ACCURACY_THRESHOLD = 10 


def get_test_predictions(model, sentences):
    input_sents = tf.data.Dataset.from_tensor_slices(
        (sentences, [0] * len(sentences))
    )
    input_sents = input_sents.map(to_feature_map).batch(1)

    predictions = model.predict(input_sents)
    print(predictions)
    return [int(round(pred[0])) for pred in predictions]

def calc_percentage_correct(actual, expected):
    comparisons = [a == b for (a, b) in itertools.product(actual, expected)]
    correct = comparisons.count(True)
    return round(100.0 * correct/len(comparisons), 2)

class TestBert(unittest.TestCase):
    def setUp(self):
        init_default_tokeniser()

    #@unittest.skip
    def test_c9(self):
        print("\n\nC9. The app collects personal identifiable information (PII).\n\n")
        c9model = tf.keras.models.load_model(C9_MODEL_PATH)
        sentences = [
            "This is a lemon tree.", # 0
            "We allow you to control your access to the data.",# 0
            "Accordingly, we have implemented transparency and access controls to help such users", # 0
            "As such, you are able to control what Personal Data you share and with whom you share it", # 0
            "Request the complete deletion of your data, including all past data sent to third-party services used for tracking and analysis, by reaching out to trust@helloclue.com", # 0
            "Users have the right, at any time, to know whether their Personal Data has been stored and can consult the Data Controller to learn about their contents and origin, to verify their accuracy or to ask for them to be supplemented, cancelled, updated or corrected, or for their transformation into anonymous format or to block any data held in violation of the law, as well as to oppose their treatment for any and all legitimate reasons.", # 0
            "Users have the right, under certain circumstances, to obtain the erasure of their Data from the Owner.", # 0
            "", # 0
            "For a better experience, while using our Service, I may require you to provide us with certain personally identifiable information.", # 1
            "Personal data including, for example, your name, e-mail address, password, and in certain instances, telephone number", # 1
            "Your membership, including your email and password, with BBS is personal and may not be transferred or used by someone else", # 1
            "We collect IP addresses provided by your browser or mobile device to deliver the service to your device. We also use the IP address to determine your approximate location for statistical and analytics purposes, and for regulatory compliance in different countries. To be clear, we do not collect your precise location.", # 1
            "information you provide us such as email, device IMEI number", # 1
            "This Personal Data may include name, photo, username and password, email address, date of birth, gender, payment information and general location data", # 1
            "Your privacy is very important to us.", # 0
            "Please be aware that the terms of this policy can be changed at any time at the sole discretion of Takalogy applications.", # 0
            "We are committed to conducting our business in accordance with these principles in order to ensure that the confidentiality of personal information is protected and maintained.", # 0
        ]

        predictions = get_test_predictions(c9model, sentences)

        expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0
        ]

        accuracy = calc_percentage_correct(predictions, expected)

        self.assertTrue(accuracy > TEST_ACCURACY_THRESHOLD)

    #@unittest.skip
    def test_c10(self):
        print("\n\nC10. The app ensures the right of access to collected information.\n\n")
        c10model = tf.keras.models.load_model(C10_MODEL_PATH)
        sentences = [
               "This is a lemon tree.", # 0
               "We allow you to control your access to the data.", # 1
               "Accordingly, we have implemented transparency and access controls to help such users", # 1
               "As such, you are able to control what Personal Data you share and with whom you share it", # 1
               "Request the complete deletion of your data, including all past data sent to third-party services used for tracking and analysis, by reaching out to trust@helloclue.com", # 1
               "Users have the right, at any time, to know whether their Personal Data has been stored and can consult the Data Controller to learn about their contents and origin, to verify their accuracy or to ask for them to be supplemented, cancelled, updated or corrected, or for their transformation into anonymous format or to block any data held in violation of the law, as well as to oppose their treatment for any and all legitimate reasons.", # 1
               "Users have the right, under certain circumstances, to obtain the erasure of their Data from the Owner.", # 1
               "", # 0
        ]

        predictions = get_test_predictions(c10model, sentences)

        expected = [
            0, 1, 1, 1, 1, 1, 1, 0
        ]

        accuracy = calc_percentage_correct(predictions, expected)

        self.assertTrue(accuracy > TEST_ACCURACY_THRESHOLD)

    #@unittest.skip
    def test_c11(self):
        print("\n\nC11. Does the app have at least one agreement with third parties?\n\n")
        c11model = tf.keras.models.load_model(C11_MODEL_PATH)
        sentences = [
            "This is a lemon tree.", # 0
            "We allow you to control your access to the data.", # 0
            "Accordingly, we have implemented transparency and access controls to help such users", # 0
            "As such, you are able to control what Personal Data you share and with whom you share it", # 0
            "Request the complete deletion of your data, including all past data sent to third-party services used for tracking and analysis, by reaching out to trust@helloclue.com", # 0
            "Users have the right, at any time, to know whether their Personal Data has been stored and can consult the Data Controller to learn about their contents and origin, to verify their accuracy or to ask for them to be supplemented, cancelled, updated or corrected, or for their transformation into anonymous format or to block any data held in violation of the law, as well as to oppose their treatment for any and all legitimate reasons.", # 0
            "Users have the right, under certain circumstances, to obtain the erasure of their Data from the Owner.", # 0
            "", # 0
            "Bruh what's up", # 0
            "While they are seemingly ubiquitous in enterprise development, they have serious drawbacks, and typically mask easily fixable deficiencies in the underlying code.", # 0
            "we may allow a third party platform to access the specific personal data you provide in order to perform the verification.", # 1
            "We do share a limited set of data that is gathered when you visit our Websites, such as cookies and pixels, with third parties in order to allow you to see tailored digital advertisements", # 1
            "We disclose various types of information for purposes of interest-based advertising, including for third party, interest-based advertising", # 1
            "We may allow you to register and pay for third-party products and services or otherwise interact with another website, mobile application, or Internet location (collectively “Third Party Sites”) through our Services, and we may collect Personal Data that you share with Third Party Sites through our Services.", # 1
            "CardioVisual may disclose to third party services certain personally identifiable information listed below", # 1
            "I want to inform you that whenever you use my Service, in a case of an error in the app I collect data and information (through third party products) on your phone called Log Data.The app does use third party services that may collect information used to identify you.", # 1
            "Among the types of Personal Data that Body By Simone collects, by itself or through third parties, there are: Camera permission, email address, first name, last name, Usage Data, Cookies, unique device identifiers for advertising", # 1
            "Data is collected by or through third parties by the app" # 1
        ]

        predictions = get_test_predictions(c11model, sentences)

        expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
        ]

        accuracy = calc_percentage_correct(predictions, expected)

        self.assertTrue(accuracy > TEST_ACCURACY_THRESHOLD)

    #@unittest.skip
    def test_c12(self):
        print("\n\nC12. The app requires users to always give their explicit consent before any action is taken.\n\n")
        c12model = tf.keras.models.load_model(C12_MODEL_PATH)
        sentences = [
            "This is a lemon tree.", # 0
            "We allow you to control your access to the data.", # 0
            "Accordingly, we have implemented transparency and access controls to help such users", # 0
            "As such, you are able to control what Personal Data you share and with whom you share it", # 0
            "Details on how to update your sharing preferences and the default settings for our Services are outlined within the sharing standard", # 0
            "", # 0
            "Bruh what's up", # 0
            "While they are seemingly ubiquitous in enterprise development, they have serious drawbacks, and typically mask easily fixable deficiencies in the underlying code.", # 0
            "You may indicate your consent in a number of ways, including, as permitted by law, ticking a box (or equivalent action) to indicate your consent", # 1
            "We operate internationally and transfer information to the United States and other countries for the purposes described in this policy.", # 0
            "we do not provide your personal data to any third party without your specific consent", # 0
            "For data collected based on your consent to receive our marketing communications: we will use such data until you opt out, withdraw consent or applicable law requires that such data is no longer used", # 1
            "In some cases, we will ask for your consent to process your Personal Data", # 1
            "You can revoke your consent to share with third-party applications or employee wellness programs using your account settings", # 1
            "Parents or guardians must consent to the use of their child’s data in accordance with the Privacy Policy for Children’s Accounts in order to create such an account.", # 1
            "If you later wish to withdraw your consent, you can delete your Fitbit account as described in the Your Rights To Access and Control Your Personal Data section.", # 1
            "To the extent that information we collect is health data or another special category of personal data subject to the GDPR, we ask for your explicit consent to process the data." # 1
        ]

        predictions = get_test_predictions(c12model, sentences)

        expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1
        ]

        accuracy = calc_percentage_correct(predictions, expected)

        self.assertTrue(accuracy > TEST_ACCURACY_THRESHOLD)

    #@unittest.skip
    def test_c24(self):
        print("\n\nC24. The app gives users control over their data.\n\n")
        c24model = tf.keras.models.load_model(C24_MODEL_PATH)
        sentences = [
            "This is a lemon tree.", # 0
            "We allow you to control your access to the data.", # 1
            "Accordingly, we have implemented transparency and access controls to help such users", # 1
            "As such, you are able to control what Personal Data you share and with whom you share it", # 1
            "Request the complete deletion of your data, including all past data sent to third-party services used for tracking and analysis, by reaching out to trust@helloclue.com", # 1
            "Users have the right, at any time, to know whether their Personal Data has been stored and can consult the Data Controller to learn about their contents and origin, to verify their accuracy or to ask for them to be supplemented, cancelled, updated or corrected, or for their transformation into anonymous format or to block any data held in violation of the law, as well as to oppose their treatment for any and all legitimate reasons.", # 1
            "Users have the right, under certain circumstances, to obtain the erasure of their Data from the Owner.", # 1
            "", # 0
        ]

        predictions = get_test_predictions(c24model, sentences)

        expected = [
            0, 1, 1, 1, 1, 1, 1, 0
        ]

        accuracy = calc_percentage_correct(predictions, expected)

        self.assertTrue(accuracy > TEST_ACCURACY_THRESHOLD)

    #@unittest.skip
    def test_c49(self):
        print("\n\nC49. The app applies appropriate measures to protect minor users in accordance with applicable legislations.\n\n")
        c49model = tf.keras.models.load_model(C49_MODEL_PATH)
        sentences = [
            "This is a lemon tree.", # 0
            "We allow you to control your access to the data.", # 0
            "Accordingly, we have implemented transparency and access controls to help such users", # 0
            "As such, you are able to control what Personal Data you share and with whom you share it", # 0
            "Request the complete deletion of your data, including all past data sent to third-party services used for tracking and analysis, by reaching out to trust@helloclue.com", # 0
            "Users have the right, at any time, to know whether their Personal Data has been stored and can consult the Data Controller to learn about their contents and origin, to verify their accuracy or to ask for them to be supplemented, cancelled, updated or corrected, or for their transformation into anonymous format or to block any data held in violation of the law, as well as to oppose their treatment for any and all legitimate reasons.", # 0
            "Users have the right, under certain circumstances, to obtain the erasure of their Data from the Owner.", # 0
            "", # 0
            "Individuals under the age of 18, or the applicable age of majority, may utilize the Products only with the involvement of a parent or legal guardian under such person's account.", # 1
            "Parents or guardians must consent to the use of their child’s data in accordance with the Privacy Policy for Children’s Accounts in order to create such an account.", # 1
            "If we learn that we have collected the personal information of a child under the relevant minimum age without parental consent, we will take steps to delete the information as soon as possible.", # 1
            "Parents who believe that their child has submitted personal information to us and would like to have it deleted may contact us at privacy@fitbit.com.", # 1
            "We appreciate the importance of taking additional measures to protect children’s privacy.", # 1
        ]

        predictions = get_test_predictions(c49model, sentences)

        expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
        ]

        accuracy = calc_percentage_correct(predictions, expected)

        self.assertTrue(accuracy > TEST_ACCURACY_THRESHOLD)

if __name__ == "__main__":
    unittest.main()