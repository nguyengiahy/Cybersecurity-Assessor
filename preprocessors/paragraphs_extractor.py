import re
import tensorflow as tf
from itertools import chain
from collections import Counter
from bs4 import BeautifulSoup
from .utils.spacy_utils import custom_extractor
from ..parameters import REMOVE_CHARACTERS
from ..models.bert.utils import to_feature_map

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
MINIMUM_EXPECTED_SEGMENTS = 6 # For situations where a policy HTML does not have enough of the above HEADING_TAGS

class ParagraphsExtractor:
    """
    First component in preprocessing pipeline. This is a pure compotent - no states

    - Extract privacy policy texts into multiple paragraphs
    - Identify topic sentence
    - Identify common keywords
    """

    def get_paragraphs(self, html):
        """Get the list of paragraphs from the HTML privacy policy

        Args:
            html (str): The raw html text

        Returns:
            (list(str)): The list of paragraph contents
        """

        # retrieve the paragraph contents from its html text
        segments = self.get_segments_by_html_tags(html)
        # chain (or concanetate) paragraphs from many `segments` into a single list of paragraphs (originally list of lists)
        paragraphs_with_tags = chain.from_iterable(
            segment.get("paragraphs") for segment in segments
        )
        # only get the text content of each paragraph
        return [
            p.get("content") for p in paragraphs_with_tags if len(p.get("content")) > 0
        ]

    def get_segments_by_html_tags(self, policy_html):
        """Get segements of a policy using html tags

        Args:
            policy_html (html): privacy policy of an app in raw html format

        Returns:
            list: a list of segments. Each segment has a header and paragraphs
        """
        segments = []
        soup = BeautifulSoup(policy_html, features="html.parser")
        for heading in soup.find_all(HEADING_TAGS):
            segment = {
                "header": {
                    "tag": heading.name,
                    "content": self.sanitise_text(heading.get_text()),
                },
                "paragraphs": self.get_paragraphs_per_segment(heading),
            }
            segments.append(segment)

        if len(segments) >= MINIMUM_EXPECTED_SEGMENTS:
            return segments

        # If not enough segments are found, treat the whole policy text as a segment
        return [
            {
            "header": {
                "tag": "",
                "content": "",
                },
            "paragraphs": [
                {
                    "tag": "",
                    "content": self.sanitise_text(soup.get_text(separator="\n", strip=True))
                }],
            }
        ]

    def get_paragraphs_per_segment(self, segmentHeader):
        """Get list of paragraphs under each segment header

        Args:
            segmentHeader (html tag): a header tag h1 or h2 or h3

        Returns:
            list: a list of paragraphs under the segment header
        """
        paragraphs = []
        for element in segmentHeader.next_siblings:
            if element is not None and element.name and element.name in HEADING_TAGS:
                break  # Found the next header, stop processing
            else:
                try:
                    para = {
                        "tag": element.name,
                        "content": self.sanitise_text(
                            element.get_text(separator="\n", strip=True)
                        ),
                    }
                    paragraphs.append(para)
                except Exception as e:
                    print(
                        "Exception {0} when parsing element {1}, header {2}: ".format(
                            e, element, segmentHeader
                        )
                    )
                    continue

        if len(paragraphs) == 0:
            # FIXME: extend the HTML segment extractor by adding a layer of logic to deal with no heading tags found.
            pass

        return paragraphs

    def get_paragraph_statistics(self, paragraph, keywords):
        """Get the statistics of a paragraph

        Args:
            paragraph (str): The paragraph content
            keywords (iter(str)): Criteria-specific keywords used to search for relevant paragraphs
        Returns:
            (dict): Paragraph Meta-statistics such as topic sentence and the keyword occurence
        """
        topic_sentence = self.get_topic_sentence(paragraph)

        paragraph_keyword_dict = self.count_keywords(paragraph, keywords)
        topic_keyword_dict = self.count_keywords(topic_sentence, keywords)

        non_topic_keyword_dict = {
            key: paragraph_keyword_dict.get(key) - topic_keyword_dict.get(key)
            for key in keywords
        }

        return {
            "content": paragraph,
            "topic_sentence": topic_sentence,
            "topic_keyword_occurence": topic_keyword_dict,
            "non_topic_keyword_occurence": non_topic_keyword_dict,
            "sentences": self.get_sentences_from_single_paragraph(paragraph)
            # TODO:
            # Add more statistics in case the pipeline needs more, i.e.,
            # "keyword_occurence": paragraph_keyword_dict,
        }

    def get_sentences_from_single_paragraph(self, paragraph):
        """Get the list of sentences of a paragraph.

        Determine the sentence boundaries by the Spacy's statistical model.

        Args:
            paragraph (str): the paragraph in text

        Returns:
            (list[str]): the list of sentences of the given paragraph.
        """
        return [sent.text for sent in custom_extractor(paragraph).sents]

    def get_sentences_from_many_paragraphs(self, paragraphs):
        """Get the list of sentences of a number of paragraphs

        Args:
            paragraphs (list(str)): list of paragraphs

        Returns:
            list: list of sentences
        """
        sentences = []
        for paragraph in paragraphs:
            sentences += self.get_sentences_from_single_paragraph(paragraph)
        return sentences

    def get_topic_sentence(self, paragraph):
        """Get the topic sentence of a paragraph

        Assuming that the topic sentence is the first sentence in the paragraph.

        Args:
            paragraph (str): the paragraph in text

        Returns:
            (str): the topic sentence of the given paragraph. If the paragraph is empty, return empty string.
        """
        sentences = self.get_sentences_from_single_paragraph(paragraph)
        return sentences[0] if len(sentences) > 0 else ""

    def tokenize_text(self, text, need_lemma=True):
        """Tokenize a piece of text into words, with consideration of different forms of a word

        Args:
            text (str): the piece of text in string

        Returns:
            (list(str)): the list of tokens
        """
        word_tokens = []

        for token in custom_extractor(text):
            if not token.is_punct:
                word_tokens.append(token.lemma_ if need_lemma else token.text)

        return word_tokens

    def count_keyword(self, text, key, need_lemma=True):
        """Count the number of occurence of a keyword with optional choice to consider the different forms of the keyword

        Args:
            text (str): the piece of meaningful text in string format
            key (str): the keyword which needs finding frequency
            need_lemma (bool): the flag to indicate whether different forms of the keyword are counted as matched (Default True).

        Returns:
            (num): the number of occurence of the keyword within the given text.
        """
        tokens = self.tokenize_text(text, need_lemma=need_lemma)
        lowercase_tokens = [token.lower() for token in tokens]
        return lowercase_tokens.count(key.lower())

    def count_keywords(self, text, keys, need_lemma=True):
        """Count the occurence of keywords with optional choice to consider the different forms of the keyword

        Args:
            text (str): the piece of meaningful text in string format
            keys (iter(str)): the keyword which needs finding frequency
            need_lemma (bool): the flag to indicate whether different forms of the keyword are counted as matched (Default True).

        Returns:
            (dict(str, num)): the number of occurence of the keyword within the given text.
        """
        # We keep every token in lowercase.
        tokens = self.tokenize_text(text, need_lemma=need_lemma)
        lowercase_tokens = [token.lower() for token in tokens]
        token_frequency_counts = Counter(lowercase_tokens)
        lowercase_keys = [key.lower() for key in keys]
        return {key: token_frequency_counts.get(key, 0) for key in lowercase_keys}

    def get_relevant_score_by_keyword_occurence(
        self,
        topic_sentence_keyword_freq,
        non_topic_sentence_keyword_freq,
        weights,
        keywords=[],
    ):
        """Calculate the relevant score of a paragraph.

        The formula for the relevance score
            score =  C_t x w_t + C_nt x w_nt

        where
            C_t: the number of matched keywords in the topic sentence.
            C_nt: the number of matched keywords in the non topic sentences.
            w_t: the weight allocated for the topic sentence.
            w_nt: the weight allocated for the non-topic sentences.

        Args:
            topic_sentence_keyword_freq (dict(str, num)): pre-calculated keyword occurence dictionary for topic sentence.
            non_topic_sentence_keyword_freq (dict(str, num)): pre-calculated keyword occurence dictionary for non-topic sentences.
            weights (list(num)): the list of weights for the relevance score formula.
                (Default to [1, 1], i.e. the keyword occurrence in the topic sentence
                will have the same impact as the keyword occurence in the non-topic sentences).
            keywords (iter(str)): collection of concerned keywords

        Returns:
            score (num): the overall relevant score of a particular paragraph against a set of keywords
        """
        if len(weights) < 2:
            normalized_weights = [0.5, 0.5]
        else:
            # need to normalize the weights as the final weights all sum up to 1.0
            denominator = sum(weights[:2])
            if denominator == 0:
                print("Error: Sum of all weights must not be 0")
                return 0
            normalized_weights = [w / denominator for w in weights[:2]]

        topic_sentence_match_count = sum(
            topic_sentence_keyword_freq.get(key, 0) for key in keywords
        )
        non_topic_sentences_match_count = sum(
            non_topic_sentence_keyword_freq.get(key, 0) for key in keywords
        )

        return (
            normalized_weights[0] * topic_sentence_match_count
            + normalized_weights[1] * non_topic_sentences_match_count
        )

    def rank_paragraphs_by_relevant_score(self, paragraphs, num_paragraphs):
        """Rank the paragraphs based on the relevant score. The score is determined by the occurrence of a certain keywords. A number of top paragraphs will be returned.
        Args:
            processed_segments (dictionary): a dictionary of segments after identifying topic sentences, keyword occurrences, and relevant score.
        Returns:
            (list): ranked paragraphs
        """
        # If all paragraphs have 0 relevance, then return all of them
        if (all(element.get("relevant_score") == 0 for element in paragraphs)):
            return paragraphs
        # Otherwise, sort them by relevance and return the top num_paragraphs
        else:
            ranked_paragraphs = sorted(
                paragraphs, key=lambda x: x.get("relevant_score"), reverse=True
            )

            # don't need to check len here, python slice already automatically checks for list len
            return ranked_paragraphs[:num_paragraphs]

    def sanitise_text(self, text):
        """Sanitise a given text by eliminating a set of special characters.

        Args:
            (string): a text content

        Returns:
            (string): a sanitised text.
        """
        return re.sub(REMOVE_CHARACTERS, " ", text)

    def process_inputs_for_bert(self, sentences):
        """Convert a list of sentences to the format BERT expect

        Args:
            sentences (list): list of sentences in string

        Returns:
            list: list of sentences but split into features expected by BERT
        """
        input_sents = tf.data.Dataset.from_tensor_slices(
            (sentences, [0] * len(sentences))
        )
        input_sents = input_sents.map(to_feature_map).batch(1)
        return input_sents