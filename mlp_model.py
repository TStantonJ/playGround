import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sequential(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(Sequential, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 100)
        self.out = nn.Linear(100, 2)

    def forward(self, input_sentence):
        num_dimensions = len(input_sentence)
        print(input_sentence.shape)
        sentence = input_sentence.clone().detach().to(DEVICE)
        embedded = self.embedding(sentence)
        hidden = self.relu(self.fc1(embedded.view(num_dimensions, sentence.size()[1],self.embedding_size)))
        hidden = self.relu(self.fc1(embedded.view(-1, self.embedding_size)))

        output = self.relu(self.fc2(hidden))
        
        output1 = self.out(output)
        #print(output1.shape)
        return output1
    
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(self.dropout)
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        return x