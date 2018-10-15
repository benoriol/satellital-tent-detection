import torch
import torch.nn as nn
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

        self.conv1_1 = nn.Conv2d(3, 16, 3)
        self.conv1_2 = nn.Conv2d(16, 16, 3)
        self.conv1_3 = nn.Conv2d(16, 16, 3)

        self.conv1_4 = nn.Conv2d(16, 16, 3)
        self.conv1_5 = nn.Conv2d(16, 16, 3)
        self.conv1_6 = nn.Conv2d(16, 16, 3)

        self.conv2_1 = nn.Conv2d(16, 16, 3)
        self.conv2_2 = nn.Conv2d(16, 16, 3)
        self.conv2_3 = nn.Conv2d(16, 16, 3)

        self.conv2_4 = nn.Conv2d(16, 16, 3)
        self.conv2_5 = nn.Conv2d(16, 16, 3)
        self.conv2_6 = nn.Conv2d(16, 16, 3)



    def forward(self, inp):

        print('inputsize', inp.size())

        out = self.conv1(inp)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)

        out = self.max22(out)

        print('max1', out.size())


        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)

        out = self.max22(out)
        print('max2', out.size())

        out = self.upsample1(out)

        print('upsampled', out.size())



        return out


if __name__ == '__main__':

    BATCH_SIZE = 1

    data = load_data()

    batch = data[0][0].unsqueeze(0)

    a = Autoencoder()

    out = a(batch)
    pass