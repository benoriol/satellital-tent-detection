"""
@author: Mar Balibrea
@license: MIT-license
This file contains auxiliar functionalities for generate.py
"""

import cv2
import numpy as np
import math
import os

def isImage(file):
    extensions = ['jpg', 'png', 'gif', 'tiff']
    return file.split('.')[-1] in extensions

def paths(parameters):

    pathForBackground = parameters.get('-bf') if '-bf' in parameters else 'data/examples/background1.jpg'
    pathForMasks = [""]
    if '-hf' in parameters:
        print(os.path)
        if os.path.isfile(parameters.get('-hf')) and isImage(parameters.get('-hf')):
            paths = [parameters.get('-hf').rsplit('/',1)[-2] + '/' + file for file in os.listdir(parameters.get('-hf').rsplit('/',1)[-2])]
            pathForTents = [parameters.get('-hf')]
            pathForMasks = pathForTents.copy()
            index = [a for a, s in enumerate(paths) if pathForTents[0].rsplit('/',1)[-1].split('.')[-2] + '-mask' in s]
            pathForMasks[0] = paths[index[0]]
        elif os.path.isdir(parameters.get('-hf')):
            paths = [parameters.get('-hf') + '/' + file for file in os.listdir(parameters.get('-hf'))]
            _pathForTents = [parameters.get('-hf') + '/' + file for file in os.listdir(parameters.get('-hf')) if (os.path.isfile(parameters.get('-hf')+'/'+file) and "mask" not in file)]
            pathForTents = [path for path in _pathForTents if isImage(path)]
            pathForMasks = pathForTents.copy()
            for i in range(len(pathForTents)):
                index = [a for a, s in enumerate(paths) if pathForTents[i].split('.')[-2] + '-mask' in s]
                pathForMasks[i] = paths[index[0]]
        else:
            pathForTents = ['data/examples/tent1.png']
            pathForMasks = ['data/examples/tent1-mask.png']
    else:
        # Path to foreground pattern
        pathForTents = ['data/examples/tent1.png']
        pathForMasks = ['data/examples/tent1-mask.png']

    return pathForTents, pathForMasks, pathForBackground

def dimensions(numberoftents):
    width = np.uint8(math.ceil(math.sqrt(numberoftents)))
    left = numberoftents - width
    i = 0
    dimension = np.zeros(numberoftents*2)
    dimension[i] = width
    while left > dimension[i]:
        width = math.ceil(math.sqrt(left))
        left = left - width
        i = i + 1
        dimension[i] = width
    dimension[i+1] = left

    return np.trim_zeros(dimension)

def crop(image, margin):

    x0 = 0
    x1 = image.shape[0] - 1

    y0 = 0
    y1 = image.shape[1] - 1

    for x in range(x1 + 1):
        if x0 == 0 and np.sum(image[x]) != 0:
            x0 = x

        if x1 == image.shape[0]-1 and np.sum(image[x1-x]) != 0:
            x1 = x1-x

        if (x0 != 0 and x1 != image.shape[0]-1):
            break

    for y in range(y1 + 1):
        if np.sum(image[:, y]) != 0 and y0 == 0:
            y0 = y

        if  y1 == image.shape[1] - 1 and np.sum(image[:, y1 - y]) != 0:
            y1 = y1 - y

        if y0 != 0 and y1 != image.shape[1]-1 and x0 != 0 and x1 != image.shape[0]-1:
            break

    x0 = math.floor(min([i for i in np.arange(x0-margin-0.5, x0, 0.5) if i >= 0]))
    y0 = math.floor(min([i for i in np.arange(y0-margin-0.5, y0, 0.5) if i >= 0]))
    x1 = math.floor(max([i for i in np.arange(x1, x1+margin+0.5, 0.5) if i < image.shape[0]]))
    y1 = math.floor(max([i for i in np.arange(y1, y1+margin+0.5, 0.5) if i < image.shape[1]]))
    return [x0, x1, y0, y1]

def getTents(parameters):

    pathForTents, pathForMasks, pathForBackground = paths(parameters)
    biggest_area = [0, 0]
    background = cv2.imread(pathForBackground)
    background_shape = background.shape
    tent_shape = np.zeros(2)
    for i in range(len(pathForTents)):

        tent_shape[0] = cv2.imread(pathForTents[i]).shape[0]
        tent_shape[1] = cv2.imread(pathForTents[i]).shape[1]
        if '-s' in parameters and float(parameters.get('-s'))*max(tent_shape[0], tent_shape[1]) < max(background_shape[0], background_shape[1]):
            tent_shape[0] = float(parameters.get('-s'))*tent_shape[0]
            tent_shape[1] = float(parameters.get('-s'))*tent_shape[1]
        else:
            tent_shape[0] = 0.8*tent_shape[0]
            tent_shape[1] = 0.8*tent_shape[1]
        if tent_shape[0]*tent_shape[1] > biggest_area[0]*biggest_area[1]:
            biggest_area = [tent_shape[0], tent_shape[1]]

    dim = dimensions(len(pathForTents)) # number of rows is .count, number of columns/row is dimensions[row]
    margin = 5
    if '-margin' in parameters:
        margin = np.uint8(parameters.get('-margin'))
    tents = np.zeros((np.uint32((biggest_area[0]+margin)*np.amax(dim)*2), np.uint32((biggest_area[1]+margin)*len(dim)*2), 3), np.uint8)
    masks = np.zeros((np.uint32((biggest_area[0]+margin)*np.amax(dim)*2), np.uint32((biggest_area[1]+margin)*len(dim)*2)), np.uint8)
    if '-m' in parameters:
        white_masks = masks.copy()
    maximum = [0, 0]
    _maximum = [0, 0]
    i = 0
    for row in range(len(dim)):
        maximum[0] = 0
        # maximum[1] el maxim actual, _maximum[1] el maxim anterior
        for column in range(np.uint8(dim[row])):
            _tent = cv2.imread(pathForTents[i])
            _mask = cv2.imread(pathForMasks[i], cv2.IMREAD_GRAYSCALE)

            if '-m' in parameters:
                _white_mask = np.ones(_mask.shape, np.uint8)*255
            if '-s' in parameters and float(parameters.get('-s'))*max(_tent.shape[0], _tent.shape[1]) < max(background.shape[0], background.shape[1]):
                relation = float(parameters.get('-s'))
            else:
                relation = 0.8
            tent = cv2.resize(_tent, (0,0), fx=relation, fy=relation)
            mask = cv2.resize(_mask, (0,0), fx=relation, fy=relation)
            if '-m' in parameters:
                white_mask = cv2.resize(_white_mask, (0,0), fx=relation, fy=relation)

            _maximum[0] = maximum[0] + tent.shape[0] + margin
            tents[maximum[0]+margin : _maximum[0], _maximum[1]+margin : _maximum[1]+tent.shape[1]+margin] = tent
            masks[maximum[0]+margin : _maximum[0], _maximum[1]+margin : _maximum[1]+tent.shape[1]+margin] = mask
            if '-m' in parameters:
                white_masks[maximum[0]+margin : _maximum[0], _maximum[1]+margin : _maximum[1]+tent.shape[1]+margin] = white_mask
            maximum[0] = _maximum[0]
            maximum[1] = _maximum[1]+tent.shape[1]+margin if _maximum[1]+tent.shape[1]+margin > maximum[1] else maximum[1]
            i = i + 1
        _maximum[1] = maximum[1]

    x0, x1, y0, y1 = crop(masks, margin)
    final_tents = tents[x0:x1, y0:y1]
    _, thresh = cv2.threshold(255-masks[x0:x1, y0:y1], 127, 255, 0)
    final_masks = cv2.bitwise_not(thresh)
    final_white_masks = white_masks[x0:x1, y0:y1] if '-m' in parameters else None

    return final_tents, final_masks, final_white_masks
