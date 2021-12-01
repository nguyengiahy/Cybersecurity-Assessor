from spacy.language import Language
import spacy
from ...parameters import SENTENCE_NONE_BOUNDARY_KEYWORDS, SENTENCE_BOUNDARY_KEYWORDS


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        # if detecting a custom, non-boundary keyword, suppress the next token from being the start of a new sentence
        if token.text.lower() in SENTENCE_NONE_BOUNDARY_KEYWORDS:
            doc[token.i + 1].is_sent_start = False

        # if detecting a custom, boundary keyword, force the next token to be the start of a new sentence
        if token.text.lower() in SENTENCE_BOUNDARY_KEYWORDS:
            doc[token.i + 1].is_sent_start = True
    return doc


custom_extractor = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
custom_extractor.enable_pipe("senter")
custom_extractor.add_pipe("set_custom_boundaries", before="senter")
