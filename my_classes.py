import torch
import numpy as NP
import itertools

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, max_len= 100):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.max_len = max_len

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X_hat = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]
        X_hat = X_hat.to_list()

        X = []
        for i in X_hat:
            tmp = i.split()
            for j in tmp:
                X.append(j)

        while len(X_hat) < self.max_len:
            X_hat.append(' ')
        return X, y

    def get_vocab(self):
        'Get range of tokens in dataset'
        # Iterate over every item in dataset and collect tokens
        unfiltered_tokens = []
        for i in range(len(self.list_IDs)):
            X, y = self.__getitem__(i)
            for item in X:
                unfiltered_tokens.append(item.split())
        
        # Get unique tokens
        unfiltered_tokens = list(itertools.chain.from_iterable(unfiltered_tokens))
        filtered = NP.array(unfiltered_tokens)

        # Filter out unwanted words

        return NP.unique(filtered)