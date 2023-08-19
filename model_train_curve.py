import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * sequence_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
            nn.Tanh())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 128
hidden_size = 64
input_size = 38
sequence_length = 10
num_layers = 5

x_data = []
y_data = []

lane_data_c = pd.read_csv('./data/lane_data_c.csv')

data_c_30 = pd.read_csv('./data/data_c30.csv')
x_data_c_30 = data_c_30.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_30 = np.concatenate((lane_data_c.values, x_data_c_30.values), axis=1)
y_data_c_30 = data_c_30.iloc[:, len(data_c_30.keys())-4:].values
for i in range(10002 - sequence_length):
    x_data.append(np.concatenate((np.concatenate((x_data_c_30[i:i + sequence_length-1], y_data_c_30[i:i + sequence_length-1]), axis=1), np.concatenate((x_data_c_30[i + sequence_length-1], [0, 0, 0, 0])).reshape((1,-1))), axis=0))
    y_data.append(y_data_c_30[i+sequence_length-1])

data_c_40 = pd.read_csv('./data/data_c40.csv')
x_data_c_40 = data_c_40.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_40 = np.concatenate((lane_data_c.values, x_data_c_40.values), axis=1)
y_data_c_40 = data_c_40.iloc[:, len(data_c_40.keys())-4:].values
for i in range(10002 - sequence_length):
    x_data.append(np.concatenate((np.concatenate((x_data_c_40[i:i + sequence_length-1], y_data_c_40[i:i + sequence_length-1]), axis=1), np.concatenate((x_data_c_40[i + sequence_length-1], [0, 0, 0, 0])).reshape((1,-1))), axis=0))
    y_data.append(y_data_c_40[i+sequence_length-1])

data_c_50 = pd.read_csv('./data/data_c50.csv')
x_data_c_50 = data_c_50.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_50 = np.concatenate((lane_data_c.values, x_data_c_50.values), axis=1)
y_data_c_50 = data_c_50.iloc[:, len(data_c_50.keys())-4:].values
for i in range(10002 - sequence_length):
    x_data.append(np.concatenate((np.concatenate((x_data_c_50[i:i + sequence_length-1], y_data_c_50[i:i + sequence_length-1]), axis=1), np.concatenate((x_data_c_50[i + sequence_length-1], [0, 0, 0, 0])).reshape((1,-1))), axis=0))
    y_data.append(y_data_c_50[i+sequence_length-1])

data_c_70 = pd.read_csv('./data/data_c70.csv')
x_data_c_70 = data_c_70.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_70 = np.concatenate((lane_data_c.values, x_data_c_70.values), axis=1)
y_data_c_70 = data_c_70.iloc[:, len(data_c_70.keys())-4:].values
for i in range(10002 - sequence_length):
    x_data.append(np.concatenate((np.concatenate((x_data_c_70[i:i + sequence_length-1], y_data_c_70[i:i + sequence_length-1]), axis=1), np.concatenate((x_data_c_70[i + sequence_length-1], [0, 0, 0, 0])).reshape((1,-1))), axis=0))
    y_data.append(y_data_c_70[i+sequence_length-1])

data_c_100 = pd.read_csv('./data/data_c100.csv')
x_data_c_100 = data_c_100.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_100 = np.concatenate((lane_data_c.values, x_data_c_100.values), axis=1)
y_data_c_100 = data_c_100.iloc[:, len(data_c_100.keys())-4:].values
for i in range(10002 - sequence_length):
    x_data.append(np.concatenate((np.concatenate((x_data_c_100[i:i + sequence_length-1], y_data_c_100[i:i + sequence_length-1]), axis=1), np.concatenate((x_data_c_100[i + sequence_length-1], [0, 0, 0, 0])).reshape((1,-1))), axis=0))
    y_data.append(y_data_c_100[i+sequence_length-1])

#x_data_no_shuffle, y_data_no_shuffle = torch.FloatTensor(x_data).to(device), torch.FloatTensor(y_data).to(device)
#temp = list(zip(x_data, y_data))
#random.shuffle(temp)
#res1, res2 = zip(*temp)
#x_data, y_data = list(res1), list(res2)
x_data = torch.FloatTensor(x_data).to(device)
y_data = torch.FloatTensor(y_data).to(device)

split = len(x_data)# * 4 // 5

x_train_seq = x_data[:split]
y_train_seq = y_data[:split]
x_test_seq = x_data[split:]
y_test_seq = y_data[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

#no_shffle = torch.utils.data.TensorDataset(x_data_no_shuffle, y_data_no_shuffle)
train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

no_shffle_train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=1, shuffle=False)
no_shffle_test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=1, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

model = RNN(input_size=input_size,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            num_layers=num_layers,
            device=device).to(device)

criterion = nn.MSELoss()

lr = 1e-3
num_epochs = 3000
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0

    for data in train_loader:

        seq, target = data # 배치 데이터.
        out = model(seq)   # 모델에 넣고,
        loss = criterion(out, target) # output 가지고 loss 구하고,

        optimizer.zero_grad() #
        loss.backward() # loss가 최소가 되게하는
        optimizer.step() # 가중치 업데이트 해주고,
        running_loss += loss.item() # 한 배치의 loss 더해주고,

    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 100 == 0:
        print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

torch.save(model.state_dict(), "D:/User_Data/Desktop/github/railroad/model/" + str(0) + ".pth")

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()


def plotting(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred.append(out.cpu().numpy().tolist()[0][0])

        #for data in test_loader:
        #    seq, target = data
        #    out = model(seq)
        #    test_pred.append(out.cpu().numpy().tolist()[0][0])

    #print(len(train_pred), len(test_pred))
    total = train_pred# + test_pred
    #print(len(total), 'length of total')
    plt.figure(figsize=(20, 10))
    plt.plot(np.ones(100) * len(train_pred), np.linspace(0, 1, 100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()


actual_y = [sublist[0] for sublist in y_data.cpu().numpy().tolist()]
plotting(no_shffle_train_loader, no_shffle_test_loader, actual_y)