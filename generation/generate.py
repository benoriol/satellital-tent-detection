import cv2
import numpy as np
import random
import math
from datetime import datetime
import os
import sys
from tqdm import tqdm

import auxiliar as aux

def generate(parameters):

    # Path to the background image
    background_path = parameters.get('-bf') if '-bf' in parameters else 'data/examples/background1.jpg'
    tent_path = parameters.get('-hf') if '-hf' in parameters else 'data/examples/tent1.png'

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

    tent, tent_mask, tent_white_mask = aux.getTents(parameters)

    max_tents_width = np.uint32(background.shape[0]/tent.shape[0])
    max_tents_height = np.uint32(background.shape[1]/tent.shape[1])

    # number of tent dispositions
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
                _dimensions = [max_tents_width, max_tents_height]
            else:
                dimensions = parameters.get('-d')
                _dimensions = np.uint32(dimensions.split('x'))
            battery_shape = _dimensions.copy() # shape of the tents disposition
        else:
            battery_shape[0] = random.choice(range(math.floor(max_tents_width/2), max_tents_width))+1
            battery_shape[1] = random.choice(range(math.floor(max_tents_height/2), max_tents_height))+1
        possible_positions = []
        x = 0
        if battery_shape[0] <= 3 and battery_shape[1] <= 3:
            for x in range(battery_shape[0]):
                for y in range(battery_shape[1]):
                    possible_positions.append([x, y])
        else:
            while x < battery_shape[0]:
                y = random.choice([0, 1, 2])
                while y < battery_shape[1]:
                    possible_positions.append([x, y])
                    y = y + 1
                x = x + random.choice([1, 2])
        positions = []

        # let's decide
        n_tents = np.uint32(parameters.get('-h')) if '-h' in parameters else None
        if n_tents == None:
            n_tents = random.choice(range(math.floor(len(possible_positions)/2), len(possible_positions)))+1
        elif n_tents > len(possible_positions):
            n_tents = len(possible_positions)
        x = 0
        while x < n_tents:
            p = random.randint(0, len(possible_positions)-1)
            positions.append(possible_positions[p])
            possible_positions.remove(possible_positions[p])
            x = x + 1

            rand = random.choice(range(0, 100))
            if rand > 25 and x < n_tents and p+1 < len(possible_positions)-1:
                if possible_positions[p][0] == possible_positions[p+1][0] and possible_positions[p][1]+1 == possible_positions[p+1][1]:
                    positions.append(possible_positions[p+1])
                    possible_positions.remove(possible_positions[p+1])
                    x = x + 1
            if rand > 50 and x < n_tents and p+2 < len(possible_positions)-1:
                if possible_positions[p+1][0] == possible_positions[p+2][0] and possible_positions[p+1][1]+1 == possible_positions[p+2][1]:
                    positions.append(possible_positions[p+1])
                    possible_positions.remove(possible_positions[p+1])
                    x = x + 1
            if rand > 75 and x < n_tents and p+3 < len(possible_positions)-1:
                if possible_positions[p+2][0] == possible_positions[p+3][0] and possible_positions[p+2][1]+1 == possible_positions[p+3][1]:
                    positions.append(possible_positions[p+1])
                    possible_positions.remove(possible_positions[p+1])
                    x = x + 1

        # let's assign
        tents = np.zeros((battery_shape[0]*tent.shape[0], battery_shape[1]*tent.shape[1], 3), np.uint8)
        mask = np.zeros((battery_shape[0]*tent.shape[0], battery_shape[1]*tent.shape[1]), np.uint8)
        if '-m' in parameters:
            white_mask = mask.copy()

        for p in positions:
            x = tent.shape[0]*p[0]
            y = tent.shape[1]*p[1]
            if random.choice(range(0, 100)) > 50:
                m = cv2.getRotationMatrix2D((np.uint32(tent.shape[1]/2), np.uint32(tent.shape[0]/2)), 180, 1)
                tent = cv2.warpAffine(tent, m, (tent.shape[1], tent.shape[0]))
                tent_mask = cv2.warpAffine(tent_mask, m, (tent_mask.shape[1], tent_mask.shape[0]))
            tents[x: x+tent.shape[0], y: y+tent.shape[1], :] = tent
            mask[x: x+tent.shape[0], y: y+tent.shape[1]] = tent_mask
            if '-m' in parameters:
                white_mask[x: x+tent.shape[0], y: y+tent.shape[1]] = tent_white_mask

        #now we rotate
        diagonal = np.uint32(np.sqrt(tents.shape[0]*tents.shape[0] + tents.shape[1]*tents.shape[1]))+1
        aux_tents = np.zeros((diagonal*2, diagonal*2, 3), np.uint8)
        aux_mask = np.zeros((diagonal*2, diagonal*2), np.uint8)

        # define initials to put tents in the middle of the auxiliar structures
        initials = [np.uint32(aux_tents.shape[0]/2-tents.shape[0]/2), np.uint32(aux_tents.shape[1]/2-tents.shape[1]/2)]
        aux_tents[initials[0]:(initials[0]+tents.shape[0]), initials[1]:(initials[1]+tents.shape[1])] = tents
        aux_mask[initials[0]:(initials[0]+tents.shape[0]), initials[1]:(initials[1]+tents.shape[1])] = mask
        if '-m' in parameters:
            aux_white_mask = np.zeros((diagonal*2, diagonal*2), np.uint8)
            aux_white_mask[initials[0]:(initials[0]+tents.shape[0]), initials[1]:(initials[1]+tents.shape[1])] = white_mask

        # rotation for batteries
        angle = np.uint32(parameters.get('-r')) if '-r' in parameters else None
        if angle == None or angle < 0 or angle > 180:
            angle = random.choice(range(0, 180))

        # Definir Rotacio
        M = cv2.getRotationMatrix2D((aux_tents.shape[0]/2, aux_tents.shape[1]/2), angle, 1)

        # Affine transformation
        rotated = cv2.warpAffine(aux_tents, M, (diagonal*2, diagonal*2))
        mask_rotated = cv2.warpAffine(aux_mask, M, (diagonal*2, diagonal*2))
        if '-m' in parameters:
            aux_white_mask_rotated = cv2.warpAffine(aux_white_mask, M, (diagonal*2, diagonal*2))

        # Getting the part of the image with tents
        x0, x1, y0, y1 = aux.crop(mask_rotated, 0)

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
        tents_big = np.zeros((background.shape[0], background.shape[1], background.shape[2]), np.uint8)
        tents_big[big_range_x[0]:big_range_x[1], big_range_y[0]:big_range_y[1], :] = cut_rotated[small_range_x[0]:small_range_x[1], small_range_y[0]:small_range_y[1]]

        # correct color
        for i in range(output.shape[2]):
            output[:, :, i] = output[:, :, i] * (1-output_mask/255)

        for i in range(tents_big.shape[2]):
            tents_big[:, :, i] = tents_big[:, :, i] * (output_mask/255)

        output = output + tents_big
        _, output_mask = cv2.threshold(output_mask,127,255,cv2.THRESH_BINARY)

        # cv2.imshow('output', output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        tent_path_ = (tent_path.split('.')[0]).split('/')[-1]
        outpath = data[0] + '/' + date + '/output/' + data[1] + '+' + tent_path_ + '-out' + str(n) + '.' + outpath_[-1]
        maskpath = data[0] + '/' + date + '/mask/' + data[1] + '+' + tent_path_ + '-out' + str(n) + '-mask' + '.' + outpath_[-1]

        cv2.imwrite(outpath, output)
        cv2.imwrite(maskpath, output_white_mask if '-m' in parameters else output_mask)

        _, thresh = cv2.threshold(255-output_mask, 127, 255, 0)
        img = cv2.bitwise_not(thresh)
        _, markers = cv2.connectedComponents(img)

        # print("\nimage " + str(n))
        # print("battery shape: " + str(battery_shape))
        # print("# groups of tents: " + str(n_tents))
        # print("# of tents: " + str(np.amax(markers)))

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
        '-m': {'-m': arg},
        '-hf': {'-hf': arg},
        '-bf': {'-bf': arg},
        '-date': {'-date': arg},
        '-margin': {'-margin': arg},
        '-help': {'-l': "number of outputs (def.: 10)",
                  '-h': "number of tents (def.: random)",
                  '-d': "battery shape dimensions (write NNxMM) (if 0x0 adjusts to background; def.: random)",
                  '-r': "rotation of all the batteries (range 0-180º) (def.: random)",
                  '-s': "scale tent factor (def.: 0.8)",
                  '-m': "output mask for input tent(s) or mask for each tent (def.: for each tent)",
                  '-hf': "input tent file (pass 1 filename or 1 folder) (def. data/examples/tent1.png)",
                  '-bf': "input background file (pass 1 filename) (def. data/examples/background1.jpg)",
                  '-date': "enter the date. mandatory.",
                  '-margin': "margin (in px) from one block to another (def. 5px)",
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
