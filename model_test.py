import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import csv


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        out = out.reshape(out.shape[0], -1) ## check
        out = self.fc(out)
        return out


x_data = []
lane_data_s = pd.read_csv('./data/lane_data_s.csv')

data_s_30 = pd.read_csv('./data/data_s30.csv')
x_data_s_30 = data_s_30.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_s_30 = np.concatenate((lane_data_s.values, x_data_s_30.values), axis=1)
y_data_s_30 = data_s_30.iloc[:, len(data_s_30.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_s_30[9997:10001], y_data_s_30[9997:10001]), axis=1),
                              np.concatenate((x_data_s_30[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_s_40 = pd.read_csv('./data/data_s40.csv')
x_data_s_40 = data_s_40.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_s_40 = np.concatenate((lane_data_s.values, x_data_s_40.values), axis=1)
y_data_s_40 = data_s_40.iloc[:, len(data_s_40.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_s_40[9997:10001], y_data_s_40[9997:10001]), axis=1),
                              np.concatenate((x_data_s_40[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_s_50 = pd.read_csv('./data/data_s50.csv')
x_data_s_50 = data_s_50.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_s_50 = np.concatenate((lane_data_s.values, x_data_s_50.values), axis=1)
y_data_s_50 = data_s_50.iloc[:, len(data_s_50.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_s_50[9997:10001], y_data_s_50[9997:10001]), axis=1),
                              np.concatenate((x_data_s_50[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_s_70 = pd.read_csv('./data/data_s70.csv')
x_data_s_70 = data_s_70.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_s_70 = np.concatenate((lane_data_s.values, x_data_s_70.values), axis=1)
y_data_s_70 = data_s_70.iloc[:, len(data_s_70.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_s_70[9997:10001], y_data_s_70[9997:10001]), axis=1),
                              np.concatenate((x_data_s_70[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_s_100 = pd.read_csv('./data/data_s100.csv')
x_data_s_100 = data_s_100.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_s_100 = np.concatenate((lane_data_s.values, x_data_s_100.values), axis=1)
y_data_s_100 = data_s_100.iloc[:, len(data_s_100.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_s_100[9997:10001], y_data_s_100[9997:10001]), axis=1),
                              np.concatenate((x_data_s_100[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

x_data_s = torch.FloatTensor(x_data).to(device)

x_data = []
lane_data_c = pd.read_csv('./data/lane_data_c.csv')

data_c_30 = pd.read_csv('./data/data_c30.csv')
x_data_c_30 = data_c_30.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_30 = np.concatenate((lane_data_c.values, x_data_c_30.values), axis=1)
y_data_c_30 = data_c_30.iloc[:, len(data_c_30.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_c_30[9997:10001], y_data_c_30[9997:10001]), axis=1),
                              np.concatenate((x_data_c_30[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_c_40 = pd.read_csv('./data/data_c40.csv')
x_data_c_40 = data_c_40.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_40 = np.concatenate((lane_data_c.values, x_data_c_40.values), axis=1)
y_data_c_40 = data_c_40.iloc[:, len(data_c_40.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_c_40[9997:10001], y_data_c_40[9997:10001]), axis=1),
                              np.concatenate((x_data_c_40[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_c_50 = pd.read_csv('./data/data_c50.csv')
x_data_c_50 = data_c_50.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_50 = np.concatenate((lane_data_c.values, x_data_c_50.values), axis=1)
y_data_c_50 = data_c_50.iloc[:, len(data_c_50.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_c_50[9997:10001], y_data_c_50[9997:10001]), axis=1),
                              np.concatenate((x_data_c_50[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_c_70 = pd.read_csv('./data/data_c70.csv')
x_data_c_70 = data_c_70.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_70 = np.concatenate((lane_data_c.values, x_data_c_70.values), axis=1)
y_data_c_70 = data_c_70.iloc[:, len(data_c_70.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_c_70[9997:10001], y_data_c_70[9997:10001]), axis=1),
                              np.concatenate((x_data_c_70[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

data_c_100 = pd.read_csv('./data/data_c100.csv')
x_data_c_100 = data_c_100.drop(columns=['Distance', 'YL_M1_B1_W1', 'YR_M1_B1_W1', 'YL_M1_B1_W2', 'YR_M1_B1_W2'])
x_data_c_100 = np.concatenate((lane_data_c.values, x_data_c_100.values), axis=1)
y_data_c_100 = data_c_100.iloc[:, len(data_c_100.keys()) - 4:].values
x_data.append(np.concatenate((np.concatenate((x_data_c_100[9997:10001], y_data_c_100[9997:10001]), axis=1),
                              np.concatenate((x_data_c_100[10001], [0, 0, 0, 0])).reshape((1, -1))), axis=0))

x_data_c = torch.FloatTensor(x_data).to(device)

out_dir = './output/answer_sample.csv'

batch_size = 128
hidden_size = 64
input_size_c = 38
input_size_s = 40
sequence_length = 15
num_layers = 5

iter_num = 1999 # or 2000

model_s = RNN(input_size=input_size_s,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            num_layers=num_layers,
            device=device).to(device)
model_c = RNN(input_size=input_size_c,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            num_layers=num_layers,
            device=device).to(device)

model_c.load_state_dict(torch.load("./model/" + str(0) + ".pth"))
model_s.load_state_dict(torch.load("./model/" + 'straight' + ".pth"))

data_y = []
"""
x_data_s_dict = [x_data_s_30, x_data_s_40, x_data_s_50, x_data_s_70, x_data_s_100]
y_data_s_dict = [y_data_s_30, y_data_s_40, y_data_s_50, y_data_s_70, y_data_s_100]

x_data_c_dict = [x_data_c_30, x_data_c_40, x_data_c_50, x_data_c_70, x_data_c_100]
y_data_c_dict = [y_data_c_30, y_data_c_40, y_data_c_50, y_data_c_70, y_data_c_100]
"""
x_data_ = []
for x in x_data_s:
    x_data_.append(x)
for x in x_data_c:
    x_data_.append(x)
x_data_dict = [x_data_s_30, x_data_s_40, x_data_s_50, x_data_s_70, x_data_s_100, x_data_c_30, x_data_c_40, x_data_c_50, x_data_c_70, x_data_c_100]
y_data_dict =[y_data_s_30, y_data_s_40, y_data_s_50, y_data_s_70, y_data_s_100, y_data_c_30, y_data_c_40, y_data_c_50, y_data_c_70, y_data_c_100]
"""
with torch.no_grad():
    for i, data in enumerate(x_data_s):
        data_ = data
        data_y.append([])
        for j in range(iter_num):
            output = model_s(torch.unsqueeze(data_, dim=0))
            data_y[i].append(output.cpu().numpy().tolist())
            y_data_s_dict[i][10002+j-1] = output.cpu().numpy()
            data_ = torch.FloatTensor(np.concatenate((np.concatenate((x_data_s_dict[i][9997+j:10001+j], y_data_s_dict[i][9997+j:10001+j]), axis=1),
                                      np.concatenate((x_data_s_dict[i][10001+j], [0, 0, 0, 0])).reshape((1, -1))), axis=0)).to(device)
    for i, data in enumerate(x_data_c):
        data_ = data
        data_y.append([])
        for j in range(iter_num):
            output = model_c(torch.unsqueeze(data_, dim=0))
            data_y[i+5].append(output.cpu().numpy().tolist())
            y_data_c_dict[i][10002+j-1] = output.cpu().numpy()
            data_ = torch.FloatTensor(np.concatenate((np.concatenate((x_data_c_dict[i][9997+j:10001+j], y_data_c_dict[i][9997+j:10001+j]), axis=1),
                                      np.concatenate((x_data_c_dict[i][10001+j], [0, 0, 0, 0])).reshape((1, -1))), axis=0)).to(device)
"""
with torch.no_grad():
    for i in range(iter_num):
        tmp = []
        for j, data in enumerate(x_data_):
            if j < 5:
                tmp.append(model_s(torch.unsqueeze(data, dim=0)).cpu().numpy().tolist())
            else:
                tmp.append(model_c(torch.unsqueeze(data, dim=0)).cpu().numpy().tolist())
            x_data_[j] = torch.FloatTensor(np.concatenate((np.concatenate((x_data_dict[j][10001-sequence_length+1+i:10001+i], y_data_dict[j][10001-sequence_length+1+i:10001+i]), axis=1),
                                      np.concatenate((x_data_dict[j][10001+i], [0, 0, 0, 0])).reshape((1, -1))), axis=0)).to(device)
        data_y.append(tmp)


data_y = np.array(data_y)
print(data_y.shape)

data_y = data_y.squeeze(axis=2)
#data_y = np.swapaxes(data_y, 0, 1)
data_y = data_y.reshape(1999, 40)
print(data_y.shape)
data_y = data_y.tolist()
a = pd.read_csv('./data/answer_sample.csv')
fields = list(a.keys())
dist = a['Distance'].values
with open(out_dir, 'w',newline='') as f:
    write = csv.writer(f)

    write.writerow(fields)
    for d, y in zip(dist, data_y):
        d_ = [d]
        d_ = d_ + y
        write.writerow(d_)
