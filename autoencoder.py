'''
    Implementing a convolutional autoencoder. https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
'''


import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
# import torchvision
# import torchvision.transforms as transforms

import cv2
from glob import glob

class DataLoader:
    def __init__(self, pattern, shuffle=False):

        self.paths_iterator = (glob(pattern))
        if shuffle:
            random.shuffle(self.paths_iterator)

        self.paths_iterator = iter(self.paths_iterator)


    def get_batch(self, size):

        src_batch = -1
        tgt_batch = -1


        for i, x in enumerate(self.paths_iterator):

            if i == 0:
                tgt_batch = torch.from_numpy(cv2.imread(x, 0) / 255).float().unsqueeze(0).unsqueeze(0).float()

                srcim = x.replace('-mask', '')
                src_batch = torch.from_numpy(cv2.imread(srcim) / 255).permute(2, 0,1).unsqueeze(0).float()

            else:

                tgtim = torch.from_numpy(cv2.imread(x, 0) / 255).float().unsqueeze(0).unsqueeze(0)
                srcim = x.replace('-mask', '')
                srcim = torch.from_numpy(cv2.imread(srcim) / 255).permute(2, 0,1).unsqueeze(0).float()

                tgt_batch = torch.cat((tgt_batch, tgtim))
                src_batch = torch.cat((src_batch, srcim))

            if i == size - 1:
                break




        return src_batch, tgt_batch

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.max22 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.upsample1 = nn.Upsample(size=(236, 353), mode='bilinear')
        self.upsample2 = nn.Upsample(size=(473, 706), mode='bilinear')

        self.conv1_1 = nn.Conv2d(3, 4, 3)
        self.conv1_2 = nn.Conv2d(4, 4, 3)
        self.conv1_3 = nn.Conv2d(4, 4, 3)

        self.conv1_4 = nn.Conv2d(4, 8, 3)
        self.conv1_5 = nn.Conv2d(8, 8, 3)
        self.conv1_6 = nn.Conv2d(8, 8, 3)

        self.conv1_7 = nn.Conv2d(8, 16, 3)
        self.conv1_8 = nn.Conv2d(16, 16, 3)

        self.conv2_1 = nn.Conv2d(16, 8, 3)
        self.conv2_2 = nn.Conv2d(8, 8, 3)
        self.conv2_3 = nn.Conv2d(8, 8, 3)

        self.conv2_4 = nn.Conv2d(8, 4, 3)
        # self.conv2_4 = nn.Conv2d(4, 4, 3)
        self.conv2_5 = nn.Conv2d(4, 4, 3)
        self.conv2_6 = nn.Conv2d(4, 1, 3)

        self.loss_mse = nn.MSELoss()

        self.sm2d = nn.Softmax2d()



    def forward(self, inp):

        # print('input energy', self.loss_mse(inp, torch.zeros(inp.size())))
        # Encoder
        out = self.conv1_1(inp)
        out = self.relu(out)
        out = self.conv1_2(out)
        # print('input energy', self.loss_mse(out, torch.zeros(out.size())))

        out = self.relu(out)
        out = self.conv1_3(out)
        out1 = self.relu(out)
        out = self.max22(out1)
        # print('input energy', self.loss_mse(out, torch.zeros(out.size())))
        # out = self.conv1_4(out)
        # out = self.relu(out)
        # out = self.conv1_5(out)
        # out = self.relu(out)
        # out = self.conv1_6(out)
        # out = self.relu(out)
        #
        # out = self.max22(out)
        #
        # out = self.conv1_7(out)
        # out = self.relu(out)
        # out = self.conv1_8(out)
        # out = self.relu(out)
        #
        # out = self.upsample1(out)
        #
        # out = self.conv2_1(out)
        # out = self.relu(out)
        # out = self.conv2_2(out)
        # out = self.relu(out)
        # out = self.conv2_3(out)
        # out = self.relu(out)

        out = self.upsample2(out)
        # print('input energy', self.loss_mse(out, torch.zeros(out.size())))
        out1 = self.upsample2(out1)

        out = torch.cat((out, out1), 1)
        out = self.conv2_4(out)
        out = self.relu(out)
        # print('input energy', self.loss_mse(out, torch.zeros(out.size())))
        out = self.conv2_5(out)
        out = self.relu(out)
        # print('input energy', self.loss_mse(out, torch.zeros(out.size())))
        out = self.conv2_6(out)
        # print('input energy', self.loss_mse(out, torch.zeros(out.size())))
        # out = self.relu(out)
        # print('out size', out.size())
        out = self.sm2d(out)
        return out

def showTensor(tensor):
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor[0]
    tensor = tensor.detach().numpy()

    cv2.imshow('image', tensor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    start = time.time()


    N_EPOCHS = 20
    BATCH_SIZE = 15

    pattern = "data/refugee-camp-before-data-out*-mask.jpg"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    autoenc = Autoencoder().to(device)

    loss_mse = nn.MSELoss()
    # loss_mse = nn.BCELoss()

    optimizer = optim.SGD(autoenc.parameters(), lr=0.5, momentum=0.5)

    for e in range(N_EPOCHS):

        data_loader = DataLoader(pattern, shuffle=True)

        src_batch, tgt_batch = data_loader.get_batch(BATCH_SIZE)

        while type(src_batch) != type(-1):

            # print('tgt batch', tgt_batch.size())
            # print('src batch', src_batch.size())

            prediction = autoenc(src_batch)

            # showTensor(tgt_batch[0])
            # showTensor(prediction[0])
            # quit()

            loss = loss_mse(prediction, tgt_batch)

            optimizer.zero_grad()

            loss.backward()

            # optimizer.step()

            src_batch, tgt_batch = data_loader.get_batch(BATCH_SIZE)

        print('epoch n: ' + str(e + 1) + '  loss:' + str(loss.item()))

        cp_name = 'models/autoencoder-epoch' + str(e+1) + '.pt'
        print('Saving checkpoint to: ' + cp_name)
        torch.save(autoenc, cp_name)

    end = time.time()

    print ('Total time elapsed:', end-start)