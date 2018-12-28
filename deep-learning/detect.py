"""
@author: Benet Oriol
@license: MIT-license
"""

import torch
from glob import glob
import random
import cv2
import datetime
import os

from convolutional_masking import Autoencoder

model = 'models/autoencoder1.pt'

TEST_FOLDER = "data/VILLAGES1"

BATCH_SHAPE = (600, 600)


class DataLoader:
    def __init__(self, metadata_path, device, shuffle=False):

        self.subpath = '/'.join(metadata_path.split('/')[:-1])

        self.metadata = [x[:-1].split(' ') for x in open(metadata_path)]

        self.device = device
        if shuffle:
            random.shuffle(self.metadata)

    def get_len(self):
        return len(self.metadata)


    def get_batch(self, size):

        src_batch = -1

        for i, x in enumerate(self.metadata):

            if i == 0:

                path = self.subpath + '/' + x[0]
                src_batch = cv2.imread(path)
                src_batch = cv2.resize(src_batch, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC) / 255
                src_batch = torch.from_numpy(src_batch).permute(2, 0, 1).unsqueeze(0).float()
                src_batch = src_batch.to(self.device)
                self.metadata = self.metadata[1:]

            else:

                path = self.subpath + '/' + x[0]
                srcim = cv2.imread(path)
                srcim = cv2.resize(srcim, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC) / 255

                srcim = torch.from_numpy(srcim).permute(2, 0, 1).unsqueeze(0).float()
                srcim = srcim.to(self.device)
                src_batch = torch.cat((src_batch, srcim))
                self.metadata = self.metadata[1:]

            if i == size - 1:
                break

        return src_batch


class DataLoaderFull:
    def __init__(self, metadata_path, device):

        self.subpath = '/'.join(metadata_path.split('/')[:-1])

        self.metadata = [x[:-1].split(' ') for x in open(metadata_path)]

        self.device = device

        print(len(self.metadata))

    def get_len(self):
        return len(self.metadata)


    def get_batch(self, size):

        src_batch = -1
        tgt_batch = -1

        for i, x in enumerate(self.metadata):

            if i == 0:
                # print(x)
                path = self.subpath + '/' + x[1]
                tgt_batch = cv2.imread(path, 0)
                tgt_batch = cv2.resize(tgt_batch, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC) / 255
                tgt_batch = torch.from_numpy(tgt_batch).float().unsqueeze(0).unsqueeze(0)
                tgt_batch = tgt_batch.to(self.device)

                path = self.subpath + '/' + x[0]
                src_batch = cv2.imread(path)
                src_batch = cv2.resize(src_batch, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC) / 255
                src_batch = torch.from_numpy(src_batch).permute(2, 0, 1).unsqueeze(0).float()
                src_batch = src_batch.to(self.device)
                self.metadata = self.metadata[1:]

            else:
                path = self.subpath + '/' + x[1]
                tgt_value = cv2.imread(path, 0)
                tgt_value = cv2.resize(tgt_value, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC) / 255
                tgt_value = torch.from_numpy(tgt_value).float().unsqueeze(0).unsqueeze(0)
                tgt_value = tgt_value.to(self.device)

                path = self.subpath + '/' + x[0]
                srcim = cv2.imread(path)
                srcim = cv2.resize(srcim, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC) / 255

                srcim = torch.from_numpy(srcim).permute(2, 0, 1).unsqueeze(0).float()
                srcim = srcim.to(self.device)
                tgt_batch = torch.cat((tgt_batch, tgt_value))
                src_batch = torch.cat((src_batch, srcim))
                self.metadata = self.metadata[1:]

            if i == size - 1:
                break

        return src_batch, tgt_batch


def error(prediction, target):
    # return torch.mean(torch.abs(prediction - target)/target)
    return torch.mean(torch.abs(prediction - target))

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    model = torch.load(model)

    BATCH_SIZE = 10

    # d = datetime.datetime.now()
    # date = str(d.year) + str(d.month) + str(d.day) + str(d.hour) + str(d.minute)

    test_metadata_path = TEST_FOLDER + '/metadata.txt'

    if not os.path.exists(TEST_FOLDER + '/results'):
        os.makedirs(TEST_FOLDER + '/results')

    results_metadata = open(TEST_FOLDER + '/results_metadata.txt', 'w')

    dataloader = DataLoader(test_metadata_path, device)
    src_batch= dataloader.get_batch(BATCH_SIZE)

    err = torch.tensor(0).float().to(device)

    j = 0

    while type(src_batch) != type(-1):

        print("batch")
        with torch.no_grad():
            output = model(src_batch)

        for i in range(output.size()[0]):
            print('out', output[i].size())

            im = output[i][0]
            im = im.cpu().detach().numpy()

            result_path = 'results/out' + str(j) + '.jpg'
            results_metadata.write(result_path + '\n')

            cv2.imwrite(TEST_FOLDER + '/' + result_path, im*255)

            j += 1

        src_batch = dataloader.get_batch(BATCH_SIZE)
