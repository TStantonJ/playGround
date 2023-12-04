import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTM_VOCAB(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, directions, dropout):
        super(LSTM_VOCAB, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.directions = directions
        self.bidirectional = True if directions == 2 else False
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=self.bidirectional, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*directions, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_sentence):
        num_dimensions = len(input_sentence)
        sentence = input_sentence.clone().detach().to(DEVICE)
        embedded = self.embedding(sentence)
        packed_output, (hidden, cell) = self.lstm(embedded.view(num_dimensions, sentence.size()[1],self.embedding_size))
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.dropout(self.fc1(hidden))
        output = self.out(output)

        return output