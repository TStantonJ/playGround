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
        #self.lstm_size = 128
        self.hidden_size = 256
        self.num_classes = 1
        self.n_layers = 2
        self.hidden_dim = 14

        # Character Embedding
        #self.embedding = nn.Embedding(self.vocab_size ,self.lstm_size)
        self.lstm = nn.LSTM(self.vocab_size,self.hidden_size, self.n_layers,  batch_first=True)#, batch_first=True)
        #self.rnn = nn.RNN(128, 256)
        self.linear1 = nn.Linear(self.hidden_size,1)
        #self.relu1= nn.ReLU()
        #self.linear2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ip , hidden):
        batch_size = ip.size(0)
        #print(ip.shape)
        #op= self.embedding(ip)
        lstm_out, hidden = self.lstm(ip, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        #op, (hn, cn) = self.lstm(ip)
        #op, hi = self.lstm(op)
        #hn = hn.view(-1, self.hidden_size)

        output = self.linear1(lstm_out)
        #output = self.relu1(output)
        #output = self.linear2(output)
        sig_out = self.sigmoid(output)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                    weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden
    

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
x_prep= [i.squeeze(1) for i in x_prep]
y_prep = [0 if i == "flicknamn" else 1 for i in raw_train[LABEL_COLUMN].tolist()]
print("X Prep",x_prep[0].shape)
X_train, X_test, y_train, y_test = train_test_split(
    x_prep,
    y_prep,
    test_size=.2,
    shuffle=True
)



#print(X_train)
#X_train_tensors = Variable(torch.Tensor(X_train))
#X_train_tensors = torch.stack(X_train[0])
X_train_tensors = torch.stack(X_train, dim=0)
print(X_train_tensors.shape)
#X_test_tensors = Variable(torch.Tensor(X_test))
X_test_tensors =  torch.stack(X_test, dim=0)


y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
#print(len(X_train),y_train_tensors.shape)

#X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
#X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
train_tensors = TensorDataset(X_train_tensors,y_train_tensors)
train_loader = DataLoader(train_tensors, shuffle=True, batch_size=batch_size, drop_last = True)
print(train_loader)
valid_loader = DataLoader(X_test_tensors, shuffle=True, batch_size=batch_size, drop_last = True)


num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

model = lstm_name(39)
model.to(device)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


ren = 0
for epoch in range(num_epochs):
    h = model.init_hidden(14)
    #print(h[0].shape)
    
    for inputs, labels in train_loader:
        h = tuple([e.data for e in h])
        #print(len(inputs),len(labels),len(h))

        #print(inputs[0].shape)
        #labels = labels.unsqueeze(1)
        ren += 5
        #print(len(inputs))
        output, h = model(inputs, h) #forward pass
        loss = criterion(output.squeeze(), labels.float())
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        
        # obtain the loss function
        #loss = criterion(output, labels)
        
        loss.backward() #calculates the loss of the loss function
        
        optimizer.step() #improve from loss, i.e backprop
        if ren % 1000 == 0:
            print("Cur loss: %1.5f" % loss.item())
            print(output[0], labels[0])
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 