import pickle
import gzip
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
device = torch.device("cuda")
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import warnings

root_dir = 'D:/Openset_signal/'
data_dir = os.path.join(root_dir, "data/")

data_list = os.listdir(data_dir)

with gzip.open(os.path.join(data_dir, data_list[0]), 'rb') as f:
    X = pickle.load(f)
with gzip.open(os.path.join(data_dir, data_list[1]), 'rb') as f:
    Y = pickle.load(f)
Y = np.array(Y)

y_0 = Y[Y == 5]
y_6413 = Y[Y == 6413]
y_3000 = Y[Y == 3000]
y_66950 = Y[Y == 66950]
y_25050 = Y[Y == 25050]
y_20052 = Y[Y == 20052]

x_0 = X[np.tile(Y == 5, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_6413 = X[np.tile(Y == 6413, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_3000 = X[np.tile(Y == 3000, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_66950 = X[np.tile(Y == 66950, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_25050 = X[np.tile(Y == 25050, 52*60).reshape(-1,52,60)].reshape(-1,52,60)
x_20052 = X[np.tile(Y == 20052, 52*60).reshape(-1,52,60)].reshape(-1,52,60)

y = np.append(y_0,y_3000, axis=0)
y = np.append(y,y_66950, axis=0)
y = np.append(y,y_25050, axis=0)
y = np.append(y,y_20052, axis=0)

x = np.append(x_0,x_3000, axis=0)
x = np.append(x,x_66950, axis=0)
x = np.append(x,x_25050, axis=0)
x = np.append(x,x_20052, axis=0)

X = x[:,10:52,:]

y_type = np.unique(y)
nSamples = [len(y[y == i]) for i in y_type]
print(nSamples)

normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights).to(device)

print(normedWeights)

class FormsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        image = self.images[idx],
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.images)

class Normal_CNN(nn.Module):

    def __init__(self, num_classes):
        super(Normal_CNN, self).__init__()

        self.conv_layer1 = self._make_conv_layer(1, 16)
        self.conv_layer2 = self._make_conv_layer(16, 32)
        self.conv_layer3 = self._make_conv_layer(32, 64)
        #         self.conv_layer4 = self._make_conv_layer(124, 256)
        #         self.conv_layer5=nn.Conv3d(256, 256, kernel_size=(1, 2, 2), padding=0)

        self.fc5 = nn.Linear(960, 128)
        self.relu = nn.LeakyReLU()
        # self.batch0=nn.BatchNorm1d(64)
        self.drop = nn.Dropout(p=0.15)
        self.fc6 = nn.Linear(128, 32)
        self.relu = nn.LeakyReLU()
        # self.batch1=nn.BatchNorm1d(124)

        self.drop = nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(32, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(5, 5), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
        )
        return conv_layer

    #     def _make_conv_layer(self, in_c, out_c):
    #         conv_layer = nn.Sequential(
    #         nn.Conv2d(in_c, out_c, kernel_size=(5, 5), stride = 1, padding=1),
    #         nn.LeakyReLU(),
    #         nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride = 1, padding=1),
    #         nn.LeakyReLU(),
    #         nn.MaxPool2d((2, 2)),
    #         )
    #         return conv_layer
    def forward(self, x):
        #        print(x.size())
        x = self.conv_layer1(x)
        #        print(x.size())
        x = self.conv_layer2(x)
        #        print(x.size())
        x = self.conv_layer3(x)
        #         print(x.size())
        #         x = self.conv_layer4(x)
        #         print(x.size())
        #         x=self.conv_layer5(x)
        #        print(x.size())
        x = x.view(x.size(0), -1)
        #        print(x.size())

        x = self.fc5(x)
        x = self.relu(x)
        # x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        # x = self.batch1(x)
        x = self.drop(x)
        x = self.fc7(x)

        return x

X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size = 0.20, random_state = 2020, stratify = y)
X_test = x_6413[:,10:52,:]
y_test = y_6413

X_test.shape
#target data standardization

X_train = X_train.transpose(0,2,1)
X_train = X_train.reshape(-1,42)

X_val = X_val.transpose(0,2,1)
X_val = X_val.reshape(-1,42)

X_test = X_test.transpose(0,2,1)
X_test = X_test.reshape(-1,42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_train = X_train_scaled.reshape(-1,60,42)

X_val_scaled = scaler.transform(X_val)
X_val = X_val_scaled.reshape(-1,60,42)

X_test_scaled = scaler.transform(X_test)
X_test = X_test_scaled.reshape(-1,60,42)

X_train = X_train.transpose(0,2,1).astype(np.float16)
X_val = X_val.transpose(0,2,1).astype(np.float16)
X_test = X_test.transpose(0,2,1).astype(np.float16)

batch_size=32

train_dataset = FormsDataset(X_train, y_train.astype(np.float16))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f'Train dataset has {len(train_data_loader)} batches of size {batch_size}')

val_dataset = FormsDataset(X_val, y_val.astype(np.float16))
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print(f'Val dataset has {len(val_data_loader)} batches of size {batch_size}')

val_dataset_2 = FormsDataset(X_val, y_val.astype(np.float16))
val_data_loader_2 = DataLoader(val_dataset_2, batch_size=1, shuffle=True)
print(f'Val dataset_2 has {len(val_data_loader_2)} batches of size {1}')

test_dataset = FormsDataset(X_test, y_test.astype(np.float16))
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f'Test dataset has {len(test_data_loader)} batches of size {1}')

############## model hyperparameter############################
model = Normal_CNN(5).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=normedWeights).to(device)
learning_rate = 0.0005 # 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 50

# Train the model
total_steps = len(train_data_loader)
total_steps_2 = len(val_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")

train_losses = []
val_losses = []
result = []
M= 100

for epoch in range(epochs):
    loss_epoch = 0
    loss_epoch_2 = 0
    for i, (images, labels) in enumerate(train_data_loader):
        model.train()

        images[0] = torch.unsqueeze(images[0], 1)
        images[0] = images[0].type(torch.FloatTensor)
        # print(type(images), len(images))
        images = images[0].to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        # Forward pass - Encoder
        out = model(images)

#       softmax = F.log_softmax(seg_outputs, dim=1)
        loss = criterion(out, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch = loss_epoch + loss.item()

    loss_epoch = loss_epoch/total_steps
    train_losses.append(loss_epoch)
#   print (f"Epoch [{epoch + 1}/{epochs}], train_Loss: {loss_epoch:4f}")

#     if epoch % 10 == 0:
#         plt.plot(train_losses)
#         plt.show()

    for  i, (images,labels) in enumerate(val_data_loader):
        model.eval()

        images[0] = torch.unsqueeze(images[0], 1)
        images[0] = images[0].type(torch.FloatTensor)
        images = images[0].to(device)
        # audios = audios.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        #Forward pass - Encoder
        out = model(images)

        #softmax = F.log_softmax(seg_outputs, dim=1)
        loss = criterion(out, labels)

        loss_epoch_2 = loss_epoch_2 + loss.item()

        softmax = F.log_softmax(out, dim=1)
        tmp_result = np.argmax(softmax.cpu().detach().numpy(), axis = 1)
        result.append(tmp_result[0])
        label.append(labels.cpu().detach().numpy()[0])

    loss_epoch_2 = loss_epoch_2/total_steps_2
    val_losses.append(loss_epoch_2)
    print (f"Epoch [{epoch + 1}/{epochs}],train_Loss: {loss_epoch:4f}, val_Loss: {loss_epoch_2:4f}")


    #50 epoch마다 train loss graph print 및 모델 저장
    if (epoch+1) % 10 == 0:
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Train')
        #os.makedirs(root_dir + '/fig/Normal_CNN/{}/{}'.format(date, data_list[2*k]), exist_ok=True)
        #plt.savefig(root_dir + '/fig/Normal_CNN/{}/{}/epoch_{}'.format(date, data_list[2*k], epoch + 1))
        plt.show()
        #plt.savefig('./result_figure/{}_segment_loss.jpg'.format(epoch+1))

    if loss_epoch_2 <= M:
        M = loss_epoch_2
        os.makedirs(root_dir + '/model/Normal_CNN/{}/all_files'.format(date), exist_ok=True)
        torch.save(model.state_dict(), root_dir + '/model/Normal_CNN/{}/all_files/epoch_{}'.format(date, epoch + 1))
        print('Epoch {} best model saved!'.format(epoch+1))
        best_epoch = epoch + 1