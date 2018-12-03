import cv2
import numpy as np
import random
import math
from datetime import datetime
import os
import sys
from tqdm import tqdm

def isImage(file):
    extensions = ['jpg', 'png', 'gif', 'tiff']
    return file.split('.')[-1] in extensions

def paths(parameters):

    pathForBackground = parameters.get('-bf') if '-bf' in parameters else 'data/refugee_camp_before_data.jpg'
    pathForMasks = [""]
    if '-hf' in parameters:
        if os.path.isfile(parameters.get('-hf')) and isImage(parameters.get('-hf')):
            pathForHouses = [parameters.get('-hf')]
            pathForMasks = [pathForHouses[0].replace('.', '-mask.') if os.path.isfile(pathForHouses[0].replace('.', '-mask.')) else pathForHouses[0]]
        elif os.path.isdir(parameters.get('-hf')):
            _pathForHouses = [parameters.get('-hf') + '/' + file for file in os.listdir(parameters.get('-hf')) if (os.path.isfile(parameters.get('-hf')+'/'+file) and "mask" not in file)]
            pathForHouses = [path for path in _pathForHouses if isImage(path)]
            pathForMasks = pathForHouses.copy()
            paths = [parameters.get('-hf') + '/' + file for file in os.listdir(parameters.get('-hf'))]
            for i in range(len(pathForHouses)):
                index = [a for a, s in enumerate(paths) if pathForHouses[i].split('.')[-2] + '-mask' in s]
                pathForMasks[i] = paths[index[0]]

        else:
            return
    else:
        # Path to foreground pattern
        pathForHouses = ['data/casa1.png']
        pathForMasks = ['data/casa1-mask.png']

    return pathForHouses, pathForMasks, pathForBackground

def dimensions(numberofhouses):
    width = np.uint8(math.ceil(math.sqrt(numberofhouses)))
    left = numberofhouses - width
    i = 0
    dimension = np.zeros(numberofhouses*2)
    dimension[i] = width
    while left > dimension[i]:
        width = math.ceil(math.sqrt(left))
        left = left - width
        i = i + 1
        dimension[i] = width
    dimension[i+1] = left

    return np.trim_zeros(dimension)

def crop(image):

    x0 = 0
    x1 = image.shape[0] - 1

    y0 = 0
    y1 = image.shape[1] - 1

    for x in range(x1+1):
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
    return [x0, x1, y0, y1]

def getHouses(parameters):

    pathForHouses, pathForMasks, pathForBackground = paths(parameters)
    biggest_area = [0, 0]
    background = cv2.imread(pathForBackground)
    background_shape = background.shape
    house_shape = np.zeros(2)
    for i in range(len(pathForHouses)):

        house_shape[0] = cv2.imread(pathForHouses[i]).shape[0]
        house_shape[1] = cv2.imread(pathForHouses[i]).shape[1]
        if '-s' in parameters and float(parameters.get('-s'))*max(house_shape[0], house_shape[1]) < max(background_shape[0], background_shape[1]):
            house_shape[0] = float(parameters.get('-s'))*house_shape[0]
            house_shape[1] = float(parameters.get('-s'))*house_shape[1]
        else:
            house_shape[0] = 0.8*house_shape[0]
            house_shape[1] = 0.8*house_shape[1]
        if house_shape[0]*house_shape[1] > biggest_area[0]*biggest_area[1]:
            biggest_area = [house_shape[0], house_shape[1]]

    dim = dimensions(len(pathForHouses)) # number of rows is .count, number of columns/row is dimensions[row]
    houses = np.zeros((np.uint32(biggest_area[0]*np.amax(dim)*2), np.uint32(biggest_area[1]*len(dim)*2), 3), np.uint8)
    masks = np.zeros((np.uint32(biggest_area[0]*np.amax(dim)*2), np.uint32(biggest_area[1]*len(dim)*2)), np.uint8)
    if '-m' in parameters:
        white_masks = np.ones((np.uint32(biggest_area[0]*np.amax(dim)*2), np.uint32(biggest_area[1]*len(dim)*2)), np.uint8)*255
    maximum = [0, 0]
    _maximum = [0, 0]
    i = 0
    for row in range(len(dim)):
        maximum[0] = 0
        # maximum[1] el maxim actual, _maximum[1] el maxim anterior
        for column in range(np.uint8(dim[row])):
            _house = cv2.imread(pathForHouses[i])
            _mask = cv2.imread(pathForMasks[i], cv2.IMREAD_GRAYSCALE)
            if '-s' in parameters and float(parameters.get('-s'))*max(_house.shape[0], _house.shape[1]) < max(background.shape[0], background.shape[1]):
                relation = float(parameters.get('-s'))
            else:
                relation = 0.8
            house = cv2.resize(_house, (0,0), fx=relation, fy=relation)
            mask = cv2.resize(_mask, (0,0), fx=relation, fy=relation)
            if '-m' in parameters:
                white_mask = cv2.resize(white_masks, (0,0), fx=relation, fy=relation)

            _maximum[0] = maximum[0] + house.shape[0] + 5
            houses[maximum[0]+5 : _maximum[0], _maximum[1]+5 : _maximum[1]+house.shape[1]+5] = house
            masks[maximum[0]+5 : _maximum[0], _maximum[1]+5 : _maximum[1]+house.shape[1]+5] = mask
            if '-m' in parameters:
                white_masks[maximum[0]+5 : _maximum[0], _maximum[1]+5 : _maximum[1]+house.shape[1]+5] = white_mask
            maximum[0] = _maximum[0]
            maximum[1] = _maximum[1]+house.shape[1]+5 if _maximum[1]+house.shape[1]+5 > maximum[1] else maximum[1]
            i = i + 1
        _maximum[1] = maximum[1]

    x0, x1, y0, y1 = crop(masks)
    final_houses = houses[x0:x1, y0:y1]
    final_masks = masks[x0:x1, y0:y1]
    final_white_masks = white_masks[x0:x1, y0:y1] if '-m' in parameters else None

    return final_houses, final_masks, final_white_masks


def generate(parameters):

    # Path to the background image
    background_path = parameters.get('-bf') if '-bf' in parameters else 'data/refugee_camp_before_data.jpg'
    house_path = parameters.get('-hf') if '-hf' in parameters else 'data/casa1.png'

    outpath_ = background_path.split('.')
    data = outpath_[0].split('/')

    date = parameters.get('-date').replace('/', '')

    if not os.path.exists(data[0] + '/' + date):
        os.makedirs(data[0] + '/' + date)

    if not os.path.exists(data[0] + '/' + date + '/mask'):
        os.makedirs(data[0] + '/' + date + '/mask')
    if not os.path.exists(data[0] + '/' + date + '/output'):
        os.makedirs(data[0] + '/' + date + '/output')

    metadatapath = data[0] + '/' + date + '/' "metadata.txt"
    mode = 'a' if os.path.isfile(metadatapath) else 'w'
    metadata = open(metadatapath, mode)

    background = cv2.imread(background_path)

    house, house_mask, house_white_mask = getHouses(parameters)

    max_houses_width = np.uint32(background.shape[0]/house.shape[0])
    max_houses_height = np.uint32(background.shape[1]/house.shape[1])

    # number of house dispositions
    n_outputs = np.uint32(parameters.get('-l')) if '-l' in parameters else 10

    for n in tqdm(range(n_outputs)):

        output = background.copy()
        output_mask = np.zeros((background.shape[0], background.shape[1]), np.uint8)
        if '-m' in parameters:
            output_white_mask = np.zeros((background.shape[0], background.shape[1]), np.uint8)

        # where will the batteries be?
        battery_shape = [1, 1]
        if '-d' in parameters:
            if parameters.get('-d') == "0x0":
                _dimensions = [max_houses_width, max_houses_height]
            else:
                dimensions = parameters.get('-d')
                _dimensions = np.uint32(dimensions.split('x'))
            battery_shape = _dimensions.copy() # shape of the houses disposition
        else:
            battery_shape[0] = random.choice(range(0, max_houses_width))+1
            battery_shape[1] = random.choice(range(0, max_houses_height))+1
        possible_positions = []
        for x in range(battery_shape[0]):
            for y in range(battery_shape[1]):
                possible_positions.append([x, y])
        positions = []

        # let's decide
        n_houses = np.uint32(parameters.get('-h')) if '-h' in parameters else None
        if n_houses == None:
            n_houses = random.choice(range(0, battery_shape[0]*battery_shape[1]))+1
        elif n_houses > battery_shape[0]*battery_shape[1]:
            n_houses = battery_shape[0]*battery_shape[1]-1
        for x in range(n_houses):
            p = random.choice(possible_positions)
            positions.append(p)
            possible_positions.remove(p)

        # let's assign
        houses = np.zeros((battery_shape[0]*house.shape[0], battery_shape[1]*house.shape[1], 3), np.uint8)
        mask = np.zeros((battery_shape[0]*house.shape[0], battery_shape[1]*house.shape[1]), np.uint8)
        if '-m' in parameters:
            white_mask = mask.copy()

        for p in positions:
            x = house.shape[0]*p[0]
            y = house.shape[1]*p[1]
            if random.choice(range(0, 100)) > 50:
                m = cv2.getRotationMatrix2D((np.uint32(house.shape[1]/2), np.uint32(house.shape[0]/2)), 180, 1)
                house = cv2.warpAffine(house, m, (house.shape[1], house.shape[0]))
                house_mask = cv2.warpAffine(house_mask, m, (house_mask.shape[1], house_mask.shape[0]))
            houses[x: x+house.shape[0], y: y+house.shape[1], :] = house
            mask[x: x+house.shape[0], y: y+house.shape[1]] = house_mask
            if '-m' in parameters:
                white_mask[x: x+house.shape[0], y: y+house.shape[1]] = house_white_mask

        #now we rotate
        diagonal = np.uint32(np.sqrt(houses.shape[0]*houses.shape[0] + houses.shape[1]*houses.shape[1]))+1
        aux_houses = np.zeros((diagonal*2, diagonal*2, 3), np.uint8)
        aux_mask = np.zeros((diagonal*2, diagonal*2), np.uint8)

        # define initials to put houses in the middle of the auxiliar structures
        initials = [np.uint32(aux_houses.shape[0]/2-houses.shape[0]/2), np.uint32(aux_houses.shape[1]/2-houses.shape[1]/2)]
        aux_houses[initials[0]:(initials[0]+houses.shape[0]), initials[1]:(initials[1]+houses.shape[1])] = houses
        aux_mask[initials[0]:(initials[0]+houses.shape[0]), initials[1]:(initials[1]+houses.shape[1])] = mask
        if '-m' in parameters:
            aux_white_mask = np.zeros((diagonal*2, diagonal*2), np.uint8)
            aux_white_mask[initials[0]:(initials[0]+houses.shape[0]), initials[1]:(initials[1]+houses.shape[1])] = white_mask

        # rotation for batteries
        angle = np.uint32(parameters.get('-r')) if '-r' in parameters else None
        if angle == None or angle < 0 or angle > 180:
            angle = random.choice(range(0, 180))

        # Definir Rotacio
        M = cv2.getRotationMatrix2D((aux_houses.shape[0]/2, aux_houses.shape[1]/2), angle, 1)

        # Affine transformation
        rotated = cv2.warpAffine(aux_houses, M, (diagonal*2, diagonal*2))
        mask_rotated = cv2.warpAffine(aux_mask, M, (diagonal*2, diagonal*2))
        if '-m' in parameters:
            aux_white_mask_rotated = cv2.warpAffine(aux_white_mask, M, (diagonal*2, diagonal*2))

        # Getting the part of the image with houses
        x0, x1, y0, y1 = crop(mask_rotated)

        cut_rotated = rotated[x0:x1, y0:y1]
        cut_mask = mask_rotated[x0:x1, y0:y1]
        if '-m' in parameters:
            cut_white_mask = aux_white_mask_rotated[x0:x1, y0:y1]

        # put rotated battery in background
        if background.shape[0] - cut_rotated.shape[0] <= 0:
            big_range_x = [0, background.shape[0]]
            small_range_x = [0, background.shape[0]]
        else:
            start_x = random.choice(range(0, background.shape[0] - cut_rotated.shape[0]))
            big_range_x = [start_x, start_x+cut_rotated.shape[0]]
            small_range_x = [0, cut_rotated.shape[0]]

        if background.shape[1] - cut_rotated.shape[1] <= 0:
            big_range_y = [0, background.shape[1]]
            small_range_y = [0, background.shape[1]]
        else:
            start_y = random.choice(range(0, background.shape[1] - cut_rotated.shape[1]))
            big_range_y = [start_y, start_y+cut_rotated.shape[1]]
            small_range_y = [0, cut_rotated.shape[1]]

        output_mask[big_range_x[0]:big_range_x[1], big_range_y[0]:big_range_y[1]] = cut_mask[small_range_x[0]:small_range_x[1], small_range_y[0]:small_range_y[1]]
        if '-m' in parameters:
            output_white_mask[big_range_x[0]:big_range_x[1], big_range_y[0]:big_range_y[1]] = cut_white_mask[small_range_x[0]:small_range_x[1], small_range_y[0]:small_range_y[1]]
        houses_big = np.zeros((background.shape[0], background.shape[1], background.shape[2]), np.uint8)
        houses_big[big_range_x[0]:big_range_x[1], big_range_y[0]:big_range_y[1], :] = cut_rotated[small_range_x[0]:small_range_x[1], small_range_y[0]:small_range_y[1]]

        # correct color
        for i in range(output.shape[2]):
            output[:, :, i] = output[:, :, i] * (1-output_mask/255)

        for i in range(houses_big.shape[2]):
            houses_big[:, :, i] = houses_big[:, :, i] * (output_mask/255)

        output = output + houses_big
        _, output_mask = cv2.threshold(output_mask,127,255,cv2.THRESH_BINARY)

        # cv2.imshow('output', output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        house_path_ = (house_path.split('.')[0]).split('/')[-1]
        outpath = data[0] + '/' + date + '/output/' + data[1] + '+' + house_path_ + '-out' + str(n) + '.' + outpath_[-1]
        maskpath = data[0] + '/' + date + '/mask/' + data[1] + '+' + house_path_ + '-out' + str(n) + '-mask' + '.' + outpath_[-1]

        cv2.imwrite(outpath, output)
        cv2.imwrite(maskpath, output_white_mask if '-m' in parameters else output_mask)

        _, thresh = cv2.threshold(255-output_mask, 127, 255, 0)
        img = cv2.bitwise_not(thresh)
        _, markers = cv2.connectedComponents(img)

        # print("\nimage " + str(n))
        # print("battery shape: " + str(battery_shape))
        # print("# groups of houses: " + str(n_houses))
        # print("# of houses: " + str(np.amax(markers)))

        metadata.write(outpath.split('/' + date + '/')[-1] + ' ' + maskpath.split('/' + date + '/')[-1] + ' ' + str(np.amax(markers)) + '\n')


    pass
    # metadata.close()

def switch(command, arg):

    switcher = {
        '-l': {'-l': arg},
        '-h': {'-h': arg},
        '-d': {'-d': arg},
        '-r': {'-r': arg},
        '-s': {'-s': arg},
        '-m': {'-m': arg},
        '-hf': {'-hf': arg},
        '-bf': {'-bf': arg},
        '-date': {'-date': arg},
        '-help': {'-l': "number of outputs (def.: 10)",
                  '-h': "number of houses (def.: random)",
                  '-d': "battery shape dimensions (write NNxMM) (if 0x0 adjusts to background; def.: random)",
                  '-r': "rotation of all the batteries (range 0-180º) (def.: random)",
                  '-s': "scale house factor (def.: 0.8)",
                  '-m': "output mask for input house(s) or mask for each house (def.: for each house)",
                  '-hf': "input house file (pass 1 filename or 1 folder) (def. data/casa1.png)",
                  '-bf': "input background file (pass 1 filename) (def. data/refugee_camp_before_data.jpg)",
                  '-date': "enter the date. mandatory.",
                  '-help': "help"
                  }
    }
    return switcher.get(command)

def printOptions():
    for key, value in switch("-help", "").items():
        print(key + ' : ' + value)


if __name__ == '__main__':
    parameters = {}
    commands = [a for a in sys.argv if '-' in a]
    if "-help" in commands:
        printOptions()
    elif "-date" not in commands:
        print("the date is mandatory")
    else:
        for command in commands:
            if len(sys.argv) - sys.argv.index(command)-1 <= 0:
                arg = "no_arg"
            else:
                if sys.argv[sys.argv.index(command)+1] in commands:
                    arg = "no_arg"
                else:
                    arg = sys.argv[sys.argv.index(command)+1]
            if command in switch(command, arg):
                parameters.update(switch(command, arg))
        generate(parameters)
