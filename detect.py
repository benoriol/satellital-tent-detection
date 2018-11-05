
import torch
from glob import glob
import random
import cv2

from autoencoder import Autoencoder

model = 'models/autoencoder-epoch5.pt'
pattern = "data/refugee-camp-before-data-out*-mask.jpg"


class DataLoader:
    def __init__(self, pattern, shuffle=False):

        self.paths_iterator = (glob(pattern))
        if shuffle:
            random.shuffle(self.paths_iterator)

        self.paths_iterator = iter(self.paths_iterator)

    def get_batch(self, size):
        src_batch = -1

        for i, x in enumerate(self.paths_iterator):
            x = x.replace('-mask', '')
            if i == 0:
                src_batch = torch.from_numpy(cv2.imread(x) / 255).permute(2, 0,1).unsqueeze(0).float()

            else:
                srcim = torch.from_numpy(cv2.imread(x) / 255).permute(2, 0,1).unsqueeze(0).float()

                src_batch = torch.cat((src_batch, srcim))

            if i == size - 1:
                break

        return src_batch


if __name__ == '__main__':
    model = torch.load(model)

    BATCH_SIZE = 1

    loader = DataLoader(pattern)

    src_batch = loader.get_batch(BATCH_SIZE)

    j = 0

    while type(src_batch) != type(-1):

        output = model(src_batch)

        for i in range(output.size()[0]):
            print('out', output[i].size())

            im = output[i][0]
            im = im.detach().numpy()



            cv2.imwrite('results/out' + str(j) + '.jpg', im*255)
            j += 1
        src_batch = loader.get_batch(BATCH_SIZE)

