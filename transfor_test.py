from lstm_model import LSTM_VOCAB
from mlp_model import Sequential
from transfor_model import Transformer, PositionalEncoding

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
#Imports

import torch
import torch.nn as nn
import torch.optim as optim
from random import sample
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from random import shuffle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Imports BERT

#from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
#from transformers import BertTokenizer, BertForSequenceClassification
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import seaborn as sns



from my_classes import Dataset


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


raw_data_path = 'news.csv'
destination_folder = 'news_data'
source_folder =destination_folder 
# Read raw data
df_raw = pd.read_csv(raw_data_path)

# Prepare columns
df_raw.drop('Unnamed: 0', axis=1, inplace=True)
df_raw['titletext'] = df_raw['title'] + " " + df_raw['text']
df_raw['titletext'] = df_raw['titletext'].str[:1000]
df_raw['label'] = (df_raw['label'] == 'FAKE').astype('int')

#Vocabulary class
max_len = df_raw['titletext'].str.len().max()
min_len = df_raw['titletext'].str.len().min()
texts = df_raw['titletext'].values
labels = df_raw['label'].values

# load pretrained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Split first
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize
input_ids_train = []
input_ids_test = []
attention_masks_train = []
attention_masks_test = []

for text in texts_train:
    # Tokenize train
    encoded_dict = tokenizer.encode_plus(text, # input texts                     
                        add_special_tokens = True, # add [CLS] at the start, [SEP] at the end
                        max_length = 128, # if input text is longer, then it gets truncated
                        padding = 'max_length', # if input text is shorter, then it gets padded to 128
                        return_attention_mask = True,   
                        return_tensors = 'tf',
                        truncation=True)

    input_ids_train.append(encoded_dict['input_ids'][0]) 
    attention_masks_train.append(encoded_dict['attention_mask'][0])

for text in texts_test:
    # Tokenize test
    encoded_dict = tokenizer.encode_plus(text,                      
                            add_special_tokens = True, 
                            max_length = 128,           
                            padding = 'max_length',
                            return_attention_mask = True,   
                            return_tensors = 'tf',
                            truncation=True)  

    input_ids_test.append(encoded_dict['input_ids'][0])
    attention_masks_test.append(encoded_dict['attention_mask'][0])

    # Convert to tensors
input_ids_train = tf.stack(input_ids_train, axis=0)
input_ids_test = tf.stack(input_ids_test, axis=0)
attention_masks_train = tf.stack(attention_masks_train, axis=0)  
attention_masks_test = tf.stack(attention_masks_test, axis=0)
#print(input_ids_train.shape, attention_masks_train.shape, labels_train.shape)

# model compilation
optimizer = Adam(learning_rate=2e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # loss='binary_crossentropy' was loss

# model training
model.fit([input_ids_train, attention_masks_train], labels_train, batch_size=64, epochs=3, validation_split=0.2)


# MODEL RUN --------------------------------------------------------------------------------------------------------------
# Model evaluation for scores
y_pred_logits = model.predict([input_ids_test, attention_masks_test]).logits
y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
y_pred_labels = tf.argmax(y_pred_logits, axis=1).numpy()

# Creating DataFrame
texts_test_series = pd.Series(texts_test, name='Text')
scores_df = pd.DataFrame(y_pred_scores, columns=['0','1'])
final_df = pd.concat([texts_test_series, scores_df], axis=1)

# Adding overall score
final_df['Overall_Score'] = final_df[['0','1']].max(axis=1)

# Save the model's architecture and weights; save the tokenizer
model.save('bert_emotion_classifier')
tokenizer.save_pretrained('bert_emotion_classifier_tokenizer')