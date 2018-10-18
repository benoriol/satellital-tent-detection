'''
    Implementing a convolutional autoencoder. https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torchvision.transforms as transforms

import cv2
from glob import glob

def load_data():
    data = []

    for d in glob("data/refugee-camp-before-data-out*-mask.jpg"):
        tgtim =torch.from_numpy(cv2.imread(d)/255).permute(2, 0, 1).float()
        srcim = d.replace('-mask', '')
        srcim = torch.from_numpy(cv2.imread(srcim)/255).permute(2, 0, 1).float()

        data.append([srcim, tgtim])

    return (data)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.max22 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.upsample1 = nn.Upsample(size=(236, 353), mode='bilinear')
        self.upsample2 = nn.Upsample(size=(473, 706), mode='bilinear')

        self.conv1_1 = nn.Conv2d(3, 16, 3)
        self.conv1_2 = nn.Conv2d(16, 16, 3)
        self.conv1_3 = nn.Conv2d(16, 16, 3)

        self.conv1_4 = nn.Conv2d(16, 32, 3)
        self.conv1_5 = nn.Conv2d(32, 32, 3)
        self.conv1_6 = nn.Conv2d(32, 32, 3)

        self.conv1_7 = nn.Conv2d(32, 64, 3)
        self.conv1_8 = nn.Conv2d(64, 64, 3)

        self.conv2_1 = nn.Conv2d(64, 32, 3)
        self.conv2_2 = nn.Conv2d(32, 32, 3)
        self.conv2_3 = nn.Conv2d(32, 32, 3)

        self.conv2_4 = nn.Conv2d(32, 16, 3)
        self.conv2_5 = nn.Conv2d(16, 16, 3)
        self.conv2_6 = nn.Conv2d(16, 1, 3)



    def forward(self, inp):

        print('inputsize', inp.size())


        # Encoder
        out = self.conv1_1(inp)
        out = self.relu(out)
        out = self.conv1_2(out)
        out = self.relu(out)
        out = self.conv1_3(out)
        out = self.relu(out)

        out = self.max22(out)

        out = self.conv1_4(out)
        out = self.relu(out)
        out = self.conv1_5(out)
        out = self.relu(out)
        out = self.conv1_6(out)
        out = self.relu(out)

        out = self.max22(out)

        out = self.conv1_7(out)
        out = self.relu(out)
        out = self.conv1_8(out)
        out = self.relu(out)

        out = self.upsample1(out)

        out = self.conv2_1(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.relu(out)
        out = self.conv2_3(out)
        out = self.relu(out)

        out = self.upsample2(out)

        out = self.conv2_4(out)
        out = self.relu(out)
        out = self.conv2_5(out)
        out = self.relu(out)
        out = self.conv2_6(out)
        out = self.relu(out)

        print('out size', out.size())
        return out


if __name__ == '__main__':

    N_EPOCHS = 5

    BATCH_SIZE = 5

    data = load_data()

    autoenc = Autoencoder()

    loss_mse = nn.MSELoss()

    optimizer = optim.SGD(autoenc.parameters(), lr=0.01, momentum=0.9)

    for e in range(N_EPOCHS):
        print('epoch n: ', e)

        n = 0
        batch = []
        for i, d in enumerate(data):

            batch.append(d)
            if (i+1) % BATCH_SIZE == 0:

                print('new batch')
                src_batch = torch.empty(BATCH_SIZE, d[0].size()[0], d[0].size()[1], d[0].size()[2])
                tgt_batch = torch.empty(BATCH_SIZE, d[1].size()[0], d[1].size()[1], d[1].size()[2])
                
                for j, x in enumerate(batch):
                    src_batch[j] = x[0]
                    tgt_batch[j] = x[1]

                prediction = autoenc(src_batch)

                loss = loss_mse(tgt_batch, prediction)

                loss.backward()

                optimizer.step()

                batch = []


        
