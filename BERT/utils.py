import os
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization

LABEL_LIST = [0, 1]
MAX_INPUT_SEQUENCES = 250
PRETRAINED_LAYER_PATH = 'saved_models/bert_raw'
CURRENT_DIR = os.path.dirname(__file__)
BERT_LAYER_PATH = os.path.join(CURRENT_DIR, PRETRAINED_LAYER_PATH)

default_tokeniser = None

# This method is needed to ensure that the bert model is not loaded in the initial main process before the 
# criteria processes are created, otherwise tensorflow does not work: https://github.com/keras-team/keras/issues/9964,
# https://stackoverflow.com/questions/56055769/load-multiple-keras-models-in-different-processes
def init_default_tokeniser():
    """
    Initialise the global 'default_tokeniser' variable of this module to ensure 1 time loading of 
    the pre-trained BERT layer.
    """
    global default_tokeniser
    if default_tokeniser is None:
        # Get pre-trained BERT layer
        BERT_LAYER = hub.KerasLayer(BERT_LAYER_PATH, trainable=True)
        VOCAB_FILE = BERT_LAYER.resolved_object.vocab_file.asset_path.numpy()
        DO_LOWER_CASE = BERT_LAYER.resolved_object.do_lower_case.numpy()
        default_tokeniser = tokenization.FullTokenizer(VOCAB_FILE, DO_LOWER_CASE)

def to_feature(text, label, label_list=LABEL_LIST, max_seq_length=MAX_INPUT_SEQUENCES, tokenizer=None):
    if tokenizer is None:
        tokenizer = default_tokeniser

    example = classifier_data_lib.InputExample(
        guid=None, text_a=text.numpy(), text_b=None, label=label.numpy())

    feature = classifier_data_lib.convert_single_example(
        0, example, label_list, max_seq_length, tokenizer)

    return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

def to_feature_map(text, label):
    input_ids, input_mask, segment_ids, label_id = tf.py_function(
        to_feature, inp=[text, label], Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

    input_ids.set_shape([MAX_INPUT_SEQUENCES])
    input_mask.set_shape([MAX_INPUT_SEQUENCES])
    segment_ids.set_shape([MAX_INPUT_SEQUENCES])
    label_id.set_shape([])

    X = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }

    return (X, label_id)