import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification,
)


DATA_COLUMN = 'namn'
LABEL_COLUMN = 'gender'
MAX_SEQUENCE_LENGTH = 512
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_LABELS = 2


# --------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize(sentences, max_length=MAX_SEQUENCE_LENGTH, padding='max_length'):
    """Tokenize using the Huggingface tokenizer
    Args:
        sentences: String or list of string to tokenize
        padding: Padding method ['do_not_pad'|'longest'|'max_length']
    """
    return tokenizer(
        sentences,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors="tf"
    )

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
raw_train = pd.read_csv("./Name_Data_set.csv")
train_data, validation_data, train_label, validation_label = train_test_split(
    raw_train[DATA_COLUMN].tolist(),
    raw_train[LABEL_COLUMN].tolist(),
    test_size=.2,
    shuffle=True
)

# --------------------------------------------------------------------------------
# Prepare TF dataset
# --------------------------------------------------------------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(tokenize(train_data)),  # Convert BatchEncoding instance to dictionary
    train_label
)).shuffle(1000).batch(BATCH_SIZE).prefetch(1)
validation_dataset = tf.data.Dataset.from_tensor_slices((
    dict(tokenize(validation_data)),
    validation_label
)).batch(BATCH_SIZE).prefetch(1)

# --------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=NUM_LABELS
)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(
    x=train_dataset,
    y=None,
    validation_data=validation_dataset,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
)
