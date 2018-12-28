
import torch
import cv2
import sys
import numpy as np

from convolutional_masking import Autoencoder

model = 'models/autoencoder1.pt'

# INPUT_IMAGE = "data/VILLAGES1/dataset/0006.jpg"
INPUT_IMAGE = sys.argv[1]

BATCH_SHAPE = (600, 600)



def getfolder(file):

    file = file.split('/')
    file = file[:-1]

    if len(file) > 0:
        return '/'.join(file)
    else:
        return '.'

if __name__ == '__main__':

    model = torch.load(model, map_location='cpu')

    path = INPUT_IMAGE
    IMG_FOLDER = getfolder(path)

    src_batch = cv2.imread(path) / 255
    src_im = np.copy(src_batch)
    src_im = np.array(src_im, dtype = np.float32)
    src_batch = cv2.resize(src_batch, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC)
    src_batch = torch.from_numpy(src_batch).permute(2, 0, 1).unsqueeze(
        0).float()

    with torch.no_grad():
        output = model(src_batch)

    for i in range(output.size()[0]):
        print('out', output[i].size())

        im = output[i][0]
        im = im.cpu().detach().numpy()
        im = np.clip(im, 0, 1)

        or_shape = src_im.shape[:2]


        im = cv2.resize(im, or_shape, interpolation=cv2.INTER_CUBIC)
        im = np.expand_dims(im, axis=2)
        im = np.concatenate((im, im, im), axis=2)
        print(im.shape, src_im.shape)
        added_image = cv2.addWeighted(src_im, 0.7, im, 0.3, 0)


        name = INPUT_IMAGE.split('/')[-1]
        result_path = 'results/' + name.split('.')[-1] + '-detected.jpg'
        overlayed = 'results/' + name.split('.')[-1] + 'detected_overlayed.jpg'

        cv2.imwrite(result_path, im * 255)
        cv2.imwrite(overlayed, added_image * 255)

        im = cv2.resize(im, (700, 700), interpolation=cv2.INTER_CUBIC)
        added_image = cv2.resize(added_image, (700, 700), interpolation=cv2.INTER_CUBIC)
        src_im = cv2.resize(src_im, (700, 700), interpolation=cv2.INTER_CUBIC)

        cv2.imshow('image', src_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image', added_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
