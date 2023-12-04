import pandas as pd
import string
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable 
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader

class lstm_name(nn.Module):
    def __init__(self, size):
        super(lstm_name, self).__init__()
        # Vars of convienence
        self.vocab_size = 39
        self.hidden_size = 128
        self.num_classes = 2
        self.middle = 64

        self.i2h = nn.Linear(self.vocab_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.vocab_size + self.hidden_size, self.middle)
        self.fc = nn.Linear(self.middle, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input , hidden):
        #print(input, hidden)
        combined = torch.cat((input.float(), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        ret = torch.zeros(1, self.hidden_size)
        return Variable(ret)
    

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


#VOCAB = ['a', 'n', 'i', 't', 's', 'r', 'd', 'k', 'l', 'c', 'e', 'm', 'g', 'x', 'v', '-', 'h', 'o', 'f', 'b', 'u', 'j', 'z', 'y', 'é', 'w', 'ö', ' ', 'å', 'p', 'q', 'ä', 'ü', 'á', 'ó', 'ë', 'ê', 'â', 'è']
VOCAB = "anitsrdklcemgxv-hofbujzyéwö åpqäüáóëêâè"
VOCAB_LEN = len(VOCAB)
batch_size = 1
print(VOCAB_LEN)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return VOCAB.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, VOCAB_LEN)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, padToMax = 0):
    if padToMax == 0:
        tensor = torch.zeros(len(line), 1, VOCAB_LEN)
    else:
        tensor = torch.zeros(padToMax, 1, VOCAB_LEN)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
# Bring in data

DATA_COLUMN = 'namn'
LABEL_COLUMN = 'gender'
raw_train = pd.read_csv("./Name_Data_set.csv")
data_preprocess = [i.lower() for i in raw_train[DATA_COLUMN].tolist()]
max_length = max([len(word) for word in data_preprocess])

x_prep =[lineToTensor(i,max_length) for i in data_preprocess]
x_prep= [i for i in x_prep]
y_prep = [0 if i == "flicknamn" else 1 for i in raw_train[LABEL_COLUMN].tolist()]
print("X Prep",x_prep[0].shape)
X_train, X_test, y_train, y_test = train_test_split(
    x_prep,
    y_prep,
    test_size=.2,
    shuffle=True
)

num_epochs = 1000 #1000 epochs
learning_rate = 0.0001 #0.001 lr

model = lstm_name(39)
model.to(device)

criterion = torch.nn.NLLLoss()   
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


ren = 0
ALL_GENDERS = ["male", "female"]
loss_log = []
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        #labels = labels.unsqueeze(1)
        ren += 1
        hidden = model.init_hidden()
        optimizer.zero_grad()
        for j in range(14):
            output, hidden = model(X_train[i][j], hidden) #forward pass
        tmp = torch.tensor([y_train[i]])
        #print(tmp)
        loss = criterion(output, tmp)
        loss.backward() #calculates the loss of the loss function
        optimizer.step() #improve from loss, i.e backprop

        topv, topi = output.data.topk(k=1, dim=1, largest=True)
        guess = ALL_GENDERS[topi[0][0]]
        gt = ALL_GENDERS[tmp.data[0]]
        if gt == guess:
                print("Correct:",guess)
        else:
            print("fail")
        # Output current stats
        if ren % 1000 == 0:
            print("Cur loss: %1.5f" % loss.item())
            topv, topi = output.data.topk(k=1, dim=1, largest=True)
            guess = ALL_GENDERS[topi[0][0]]
            gt = ALL_GENDERS[tmp.data[0]]
            if gt == guess:
                print("Correct:",guess)

    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 