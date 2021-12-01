import os
import sys
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import sub-modules from the downloaded tensorflowlib folder
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization

CURRENT_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(CURRENT_DIR, "data/C9sentence.csv") # change dataset suitable for your criteria

# Load dataset
df = pd.read_csv(FILE_PATH)

# Create train and valid dataframe
train_df, remaining = train_test_split(
    df, random_state=42, train_size=0.75, stratify=df.target.values)
valid_df, test_df = train_test_split(
    remaining, random_state=42, train_size=0.9, stratify=remaining.target.values)

train_data = tf.data.Dataset.from_tensor_slices(
    (train_df['question_text'].values, train_df['target'].values))
valid_data = tf.data.Dataset.from_tensor_slices(
    (valid_df['question_text'].values, valid_df['target'].values))
test_data = tf.data.Dataset.from_tensor_slices(
    (test_df['question_text'].values, test_df['target'].values))

# Download BERT layer
# Hyperparameters
label_list = [0, 1]  # Label categories
# maximum length of (token) input sequences. BERT allows max = 512 tokens
max_seq_length = 128
train_batch_size = 32

PRETRAINED_LAYER_PATH = 'saved_models/bert_raw'
CURRENT_DIR = os.path.dirname(__file__)
BERT_LAYER_PATH = os.path.join(CURRENT_DIR, PRETRAINED_LAYER_PATH)

# Get pre-trained BERT layer
bert_layer = hub.KerasLayer(BERT_LAYER_PATH, trainable=True)

# Tokenizing the input is done by the provided BERT tokenizer
# Instantiate BERT tokenizer:
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# check if the imported BERT model is case sensitive version or not
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# This function converts row data to input features and label
def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
    example = classifier_data_lib.InputExample(
        guid=None, text_a=text.numpy(), text_b=None, label=label.numpy())

    feature = classifier_data_lib.convert_single_example(
        0, example, label_list, max_seq_length, tokenizer)

    return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)


# This is a wrapper function
def to_feature_map(text, label):
    input_ids, input_mask, segment_ids, label_id = tf.py_function(
        to_feature, inp=[text, label], Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    label_id.set_shape([])

    X = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }

    return (X, label_id)

# Create input pipeline
# train
train_data = (train_data.map(to_feature_map,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)  # num_parallel_calls: how much do we want to run parallel preprocessing
              .shuffle(1000)
              .batch(32, drop_remainder=True)
              .prefetch(tf.data.experimental.AUTOTUNE))

# valid
valid_data = (valid_data.map(to_feature_map,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)  # num_parallel_calls: how much do we want to run parallel preprocessing
              .batch(32, drop_remainder=True)
              .prefetch(tf.data.experimental.AUTOTUNE))

# Building the model
def create_model():
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_type_ids")

    pooled_output, sequence_output = bert_layer(
        [input_word_ids, input_mask, input_type_ids])

    # dropout regularization to prevent overfitting
    drop = tf.keras.layers.Dropout(0.4)(pooled_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(
        drop)   # only 1 sigmoid unit because we doing binary classification

    model = tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=output
    )

    return model

# Compile model
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.binary_accuracy])

# Train model
epochs = 1  # no. training iterations - keep it as 1 to save time for now.
model.fit(train_data,
          validation_data=valid_data,
          epochs=epochs,
          verbose=1)
          # callbacks=[checkpoint_callback])  # pass callback to training

# Save model
model.save('saved_models/c9') # Change c9 to your criteria
