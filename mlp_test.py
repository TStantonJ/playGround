from lstm_model import LSTM_VOCAB
from mlp_model import Sequential, MLP

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


#Vocabulary class

UNK_TOKEN = 9
class Vocab:
    def __init__(self):
        self.word2id = {"__unk__": UNK_TOKEN}
        self.id2word = {UNK_TOKEN: "__unk__"}
        self.n_words = 1

        self.tag2id = {"FAKE": 0, "REAL": 1}
        self.id2tag = {0: "FAKE", 1: "REAL"}

    def index_words(self, words):
        word_indexes = [self.index_word(w) for w in words]
        return word_indexes

    def index_tags(self, tag):
        tag_index = self.tag2id[tag]
        return tag_index

    def index_word(self, w):
        if w not in self.word2id:
            self.word2id[w] = self.n_words
            self.id2word[self.n_words] = w
            self.n_words += 1
        return self.word2id[w]
    

vocab = Vocab()
def prepare_data(data, vocab, input_field):
    data_sequences = []

    for _, row in data.iterrows():
        words = row[input_field].split()
        tags = row["label"]
        word_ids = torch.tensor(vocab.index_words(words), dtype=torch.float).to(DEVICE)
        tag_ids = torch.tensor(vocab.index_tags(tags), dtype=torch.long).to(DEVICE)
        data_sequences.append([word_ids, tag_ids])

    return data_sequences, vocab

#Create data sequnce

sequences, vocab = prepare_data(df_raw, vocab, "titletext")
x = [i[0] for i in sequences]
y = [i[1] for i in sequences]

# pad sentences to use batches
padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
x = [i for i in padded_x]

# Number of unique words

print(vocab.n_words)



# Split data to train, validation and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
test_sequences = list(zip(x_test,y_test))
test_sequences = [list(x) for x in test_sequences]
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
train_sequences = list(zip(x_train,y_train))
train_sequences = [list(x) for x in train_sequences]
val_sequences = list(zip(x_val,y_val))
val_sequences = [list(x) for x in val_sequences]

def evaluate(eval_sequences, batch_size):
    eval_loader = DataLoader(eval_sequences, batch_size=batch_size, shuffle=True)
    preds = []
    tags = []
    with torch.no_grad():
        for words, tag in eval_loader:
            preds.append(model(words).argmax(dim=1).cpu().data.numpy()) 
            tags.append(tag.cpu().data.numpy())
    preds = np.concatenate(preds).ravel()
    tags = np.concatenate(tags).ravel()
    accuracy = (preds == tags).sum() / len(tags) * 100
    return accuracy

def train_loop(model, n_epochs, batch_size, train_set, test_set):
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for e in range(1, n_epochs + 1):
        count = 0
        for words, tags in  iter(train_loader):
            model.zero_grad()
            seq_len = len(words)
            sentence_loss = 0
            output = model(words)
            #print(output.shape, tags.shape)
            #print(words.shape, output.shape)
            sentence_loss = criterion(output, tags)
            sentence_loss.backward()
            optimizer.step()
            if count % 100 == 0:
                print(f"Epoch #{e}, Batch: {count},  Loss: {sentence_loss}")
            count += 1


        train_accuracy = evaluate(train_set, batch_size)
        print(f"Epoch {e}, Training Accuracy: {train_accuracy}%")

        test_accuracy = evaluate(test_set, batch_size)
        print(f"Epoch {e}, Validation Accuracy: {test_accuracy}%")


# See what more important, title or text

# PreProcess data
title_sequences, vocab = prepare_data(df_raw, vocab, "title")
title_x = [i[0] for i in title_sequences]
title_y = [i[1] for i in title_sequences]

# pad sentences to use batches
title_padded_x = torch.nn.utils.rnn.pad_sequence(title_x, batch_first=True)
title_x = [i for i in title_padded_x]

title_x_train, title_x_test, title_y_train, title_y_test = train_test_split(title_x, title_y, test_size=0.2, random_state=42)
title_test_sequences = list(zip(title_x_test,title_y_test))
title_test_sequences = [list(x) for x in title_test_sequences]
title_train_sequences = list(zip(title_x_train,title_y_train))
title_train_sequences = [list(x) for x in title_train_sequences]

input_size = vocab.n_words
embedding_size = 300
hidden_sizes = [300, 500]
output_size = len(vocab.id2tag)
n_layers = [2,3]
directions = [2]
n_epochs = 50
dropout = 0.2
batch_size = 32
result_df = pd.DataFrame([])
print(output_size)

print("Train only on Titles SEQ")

caption = f"hidden_size - {300},  dropout {dropout}"
#model = Sequential(input_size, embedding_size, 300, output_size)
hidden_sizes = [300, 500, 300]   # List of hidden layer sizes
output_size = len(vocab.id2tag)  # Output size

# Example instantiation and usage:
model = MLP(53, hidden_sizes, output_size, dropout=0.2)
train_loop(model, n_epochs, batch_size, title_train_sequences, title_test_sequences)
train_accuracy = evaluate(title_train_sequences, batch_size)
test_accuracy = evaluate(title_test_sequences, batch_size)
temp_df = pd.DataFrame([[train_accuracy, test_accuracy]], index=[caption], columns=["training_accuracy", "test_accuracy"])
pd.concat([result_df, temp_df], ignore_index=True)