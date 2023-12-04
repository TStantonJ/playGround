import pandas as pd
import string
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification,
)


from torch.nn.functional import pad



DATA_COLUMN = 'namn'
LABEL_COLUMN = 'gender'
MAX_SEQUENCE_LENGTH = 512
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_LABELS = 2
VOCAB = ['a', 'n', 'i', 't', 's', 'r', 'd', 'k', 'l', 'c', 'e', 'm', 'g', 'x', 'v', '-', 'h', 'o', 'f', 'b', 'u', 'j', 'z', 'y', 'é', 'w', 'ö', ' ', 'å', 'p', 'q', 'ä', 'ü', 'á', 'ó', 'ë', 'ê', 'â', 'è']

raw_train = pd.read_csv("./Name_Data_set.csv")
train_data, validation_data, train_label, validation_label = train_test_split(
    raw_train[DATA_COLUMN].tolist(),
    raw_train[LABEL_COLUMN].tolist(),
    test_size=.2,
    shuffle=True
)


def get_char_wise_word_embeddings(labels, vocab: str = None, embed_size: int = 256, kernel_size: int = 3):
    
    # setting up the vocab
    if not vocab:
        vocab = [char for char in string.printable]
        
    char_to_idx_map = {char: idx for idx, char in enumerate(vocab)}
    ohe_characters = torch.eye(n=len(vocab))

    words = labels

    max_length = max([len(word) for word in words])
    
    ohe_words = torch.empty(size=(0, len(vocab), max_length))

    for word in words:
        idx_representation = [char_to_idx_map[char] for char in word]
        ohe_representation = ohe_characters[idx_representation].T
        padded_ohe_representation = pad(ohe_representation, (0, max_length-len(word)))
        ohe_words = torch.cat((ohe_words, padded_ohe_representation.unsqueeze(dim=0)))
        
    # Initialising the layers
    convolution_layer = nn.Conv1d(in_channels=len(vocab), out_channels=embed_size, kernel_size=kernel_size, bias=False)
    activation_layer = nn.Tanh()
    max_pooling_layer = nn.MaxPool1d(kernel_size=max_length-kernel_size+1)
        
    conv_out = convolution_layer(ohe_words)
    activation_out = activation_layer(conv_out)
    max_pool_out = max_pooling_layer(activation_out)
    print(max_pool_out)
    return max_pool_out.squeeze()



""" lab = raw_train['namn']
print(len(raw_train['namn']))

vocab = []
for i in lab:
    i = i.lower()
    #print(i)
    spl = [*i]
    #print(spl)
    for j in spl:
        if j not in vocab:
            vocab.append(j)

print(vocab)  """

fixed_raw_train = [i.lower() for i in raw_train['namn']]
ten = get_char_wise_word_embeddings(fixed_raw_train,VOCAB)
print(ten.shape)