import cv2
import numpy as np
import random
import math
from datetime import datetime
import os
import sys

def getHouses(pathForHouses, pathForBackground):
    biggest_area = [0, 0]
    for house in pathForHouses:
        shape = cv2.imread(house_path[0]).shape
        if shape[0]*shape[1] > biggest_area
            biggest_area
            # ESTIC FENT AIXO PER ENTREAR MES D'UNA IMATGE (ADAPTAR A LA MES GRAN LES PETITES)

def generate(parameters):

    house_path = [""]
    if '-hf' in parameters:
        if os.path.isfile(parameters.get('-hf')):
            house_path = [parameters.get('-hf')]
        elif os.path.isdir(parameters.get('-hf')):
            files = [file for file in os.listdir('data/') if os.path.isfile(file)]
            for file in files:
                house_path.append(file)
        else:
            return
    else:
        # Path to foreground pattern
        house_path = ['data/casa1.png']
        house_mask_path = ['data/casa1-mask.png']

    # Path to the background image
    background_path = 'data/refugee-camp-before-data.jpg'


    outpath_ = background_path.split('.')
    data = outpath_[0].split('/')

    dt = datetime.now()
    date = f'{dt:%Y%m%d%H%M}'

    if not os.path.exists(data[0] + '/' + date):
        os.makedirs(data[0] + '/' + date)

    if not os.path.exists(data[0] + '/' + date + '/mask'):
        os.makedirs(data[0] + '/' + date + '/mask')
    if not os.path.exists(data[0] + '/' + date + '/output'):
        os.makedirs(data[0] + '/' + date + '/output')

    metadatapath = data[0] + '/' + date + '/' "metadata.txt"
    metadata = open(metadatapath, "w")

    background = cv2.imread(background_path)
    _house = cv2.imread(house_path[0])
    _house_mask = cv2.imread(house_mask_path[0], cv2.IMREAD_GRAYSCALE)

    factor = math.sqrt(_house.shape[0]*_house.shape[1]/(background.shape[0]*background.shape[1]))
    print("\nsize factor (house/background): " + str(factor))
    if '-s' in parameters:
        _relation = np.uint32(parameters.get('-s'))
        relation = _relation if _relation <= min(1, factor) else 0.2
    else:
        relation = 0.2 if factor > 0.2 else factor
    resize = relation/factor
    scale = 1/resize if 1/resize < 1 else resize
    print("final scaling: " + str(scale))
    house = cv2.resize(_house, (0,0), fx=scale, fy=scale)
    house_mask = cv2.resize(_house_mask, (0,0), fx=scale, fy=scale)

    max_houses_width = np.uint32(background.shape[0]/house.shape[0])
    max_houses_height = np.uint32(background.shape[1]/house.shape[1])

    # number of house dispositions
    n_outputs = np.uint32(parameters.get('-l')) if '-l' in parameters else 10

    for n in range(n_outputs):

        output = background.copy()
        output_mask = np.zeros((background.shape[0], background.shape[1]), np.uint8)

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
            battery_shape[0] = random.choice(range(2, max_houses_width))
            battery_shape[1] = random.choice(range(2, max_houses_height))
        possible_positions = []
        for x in range(battery_shape[0]):
            for y in range(battery_shape[1]):
                possible_positions.append([x, y])
        positions = []

        # let's decide
        n_houses = np.uint32(parameters.get('-h')) if '-h' in parameters else None
        if n_houses == None:

            n_houses = random.choice(range(1, battery_shape[0]*battery_shape[1]-1))
        elif n_houses > battery_shape[0]*battery_shape[1]:
            n_houses = battery_shape[0]*battery_shape[1]-1
        for x in range(n_houses):
            p = random.choice(possible_positions)
            positions.append(p)
            possible_positions.remove(p)

        # let's assign
        houses = np.zeros((battery_shape[0]*house.shape[0], battery_shape[1]*house.shape[1], 3), np.uint8)
        mask = np.zeros((battery_shape[0]*house.shape[0], battery_shape[1]*house.shape[1]), np.uint8)

        for p in positions:
            x = house.shape[0]*p[0]
            y = house.shape[1]*p[1]

            houses[x: x+house.shape[0], y: y+house.shape[1], :] = house
            mask[x: x+house.shape[0], y: y+house.shape[1]] = house_mask

        #now we rotate
        diagonal = np.uint32(np.sqrt(houses.shape[0]*houses.shape[0] + houses.shape[1]*houses.shape[1]))+1
        aux_houses = np.zeros((diagonal*2, diagonal*2, 3), np.uint8)
        aux_mask = np.zeros((diagonal*2, diagonal*2), np.uint8)
        # define initials to put houses in the middle of the auxiliar structures
        initials = [np.uint32(aux_houses.shape[0]/2-houses.shape[0]/2), np.uint32(aux_houses.shape[1]/2-houses.shape[1]/2)]
        aux_houses[initials[0]:(initials[0]+houses.shape[0]), initials[1]:(initials[1]+houses.shape[1])] = houses
        aux_mask[initials[0]:(initials[0]+houses.shape[0]), initials[1]:(initials[1]+houses.shape[1])] = mask

        # rotation for batteries
        angle = np.uint32(parameters.get('-r')) if '-r' in parameters else None
        if angle == None or angle < -90 or angle > 90:
            angle = random.choice(range(-90, 90))

        # Definir Rotacio
        M = cv2.getRotationMatrix2D((aux_houses.shape[0]/2, aux_houses.shape[1]/2), angle, 1)

        # Affine transformation
        rotated = cv2.warpAffine(aux_houses, M, (diagonal*2, diagonal*2))
        mask_rotated = cv2.warpAffine(aux_mask, M, (diagonal*2, diagonal*2))

        # Getting the part of the image with houses
        x0 = 0
        x1 = mask_rotated.shape[0] - 1

        y0 = 0
        y1 = mask_rotated.shape[1] - 1

        for x in range(x1+1):
            if x0 == 0 and np.sum(mask_rotated[x]) != 0:
                x0 = x

            if x1 == mask_rotated.shape[0]-1 and np.sum(mask_rotated[x1-x]) != 0:
                x1 = x1-x

            if (x0 != 0 and x1 != mask_rotated.shape[0]-1):
                break

        for y in range(y1 + 1):
            if np.sum(mask_rotated[:, y]) != 0 and y0 == 0:
                y0 = y

            if  y1 == mask_rotated.shape[1] - 1 and np.sum(mask_rotated[:, y1 - y]) != 0:
                y1 = y1 - y

            if y0 != 0 and y1 != mask_rotated.shape[1]-1 and x0 != 0 and x1 != mask_rotated.shape[0]-1:
                break

        cut_rotated = rotated[x0:x1, y0:y1]
        cut_mask = mask_rotated[x0:x1, y0:y1]

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
        houses_big = np.zeros((background.shape[0], background.shape[1], background.shape[2]), np.uint8)
        houses_big[big_range_x[0]:big_range_x[1], big_range_y[0]:big_range_y[1], :] = cut_rotated[small_range_x[0]:small_range_x[1], small_range_y[0]:small_range_y[1]]

        # correct color
        for i in range(output.shape[2]):
            output[:, :, i] = output[:, :, i] * (1-output_mask/255)

        for i in range(houses_big.shape[2]):
            houses_big[:, :, i] = houses_big[:, :, i] * (output_mask/255)

        output = output + houses_big
        # _, output_mask = cv2.threshold(output_mask,127,255,cv2.THRESH_BINARY)

        # cv2.imshow('output', output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        outpath = data[0] + '/' + date + '/output/' + data[1] + '-out' + str(n) + '.' + outpath_[-1]
        maskpath = data[0] + '/' + date + '/mask/' + data[1] + '-out' + str(n) + '-mask' + '.' + outpath_[-1]

        cv2.imwrite(outpath, output)
        cv2.imwrite(maskpath, output_mask)

        _, thresh = cv2.threshold(255-output_mask, 127, 255, 0)
        img = cv2.bitwise_not(thresh)
        _, markers = cv2.connectedComponents(img)

        print("\nimage " + str(n))
        print("battery shape: " + str(battery_shape))
        print("# groups of houses: " + str(n_houses))
        print("# of houses: " + str(np.amax(markers)))

        metadata.write(outpath.split('/' + date + '/')[-1] + ' ' + maskpath.split('/' + date + '/')[-1] + ' ' + str(np.amax(markers)) + '\n')


    pass
    metadata.close()

def switch(command, arg):

    switcher = {
        '-l': {'-l': arg},
        '-h': {'-h': arg},
        '-d': {'-d': arg},
        '-r': {'-r': arg},
        '-s': {'-s': arg},
        '-in': {'-in': arg},
        '-help': {'-l': "number of outputs (def.: 10)",
                  '-h': "number of houses (def.: random)",
                  '-d': "battery shape dimensions (write NNxMM) (if 0x0 adjusts to background; def.: random)",
                  '-r': "rotation of all the batteries (range -+90º) (def.: random)",
                  '-s': "scale house factor (<=1) (def.: 0.15)",
                  '-hf': "input house file(s) (pass 1 filename or 1 folder) (def. data/casa1.png)",
                  '-bf': "input background file (pass 1 filename) (def. data/refugee-camp-before-data.jpg)",
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
    else:
        for command in commands:
            if sys.argv[sys.argv.index(command)+1] in commands:
                arg = "no_arg"
            else:
                arg = sys.argv[sys.argv.index(command)+1]
            if command in switch(command, arg):
                parameters.update(switch(command, arg))
        generate(parameters)
