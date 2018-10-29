import cv2
import numpy as np
import random
<<<<<<< HEAD
from datetime import datetime
import os

=======
# KK prova
>>>>>>> master
# Path to the background image
#hola
background_path = 'data/refugee-camp-before-data.jpg'
# Path to foreground pattern
house_path = 'data/casa1.jpg'
<<<<<<< HEAD
house_mask_path = 'data/casa1-mask.png'
=======
# Mininum and maximum number of rows and columns
# battery_boundaries = [[3, 7], [2, 5]]
>>>>>>> master

battery_shape = [10, 15] # shape of the houses disposition
n_houses = 80 # number of houses per battery
n_outputs = 10 # number of house dispositions

background = cv2.imread(background_path)
house = cv2.imread(house_path)
house_mask = cv2.imread(house_mask_path, cv2.IMREAD_GRAYSCALE)

dt = datetime.now()
date = f'{dt:%Y%m%d%H%M}'

for n in range(n_outputs):

    output = background.copy()
    output_mask = np.zeros((background.shape[0], background.shape[1]), np.uint8)

    # where will the batteries be?
    possible_positions = []
    for x in range(battery_shape[0]):
        for y in range(battery_shape[1]):
            possible_positions.append([x, y])
    positions = []

    # let's decide
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

    rotation = random.choice(range(-90, 90))

    # Definir Rotacio
    M = cv2.getRotationMatrix2D((aux_houses.shape[0]/2, aux_houses.shape[1]/2), rotation, 1)

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

    # cv2.imshow('hola', output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

<<<<<<< HEAD
=======
        pass

>>>>>>> master
    outpath_ = background_path.split('.')
    data = outpath_[0].split('/')

    if not os.path.exists(data[0] + '/' + date):
        os.makedirs(data[0] + '/' + date)

    if not os.path.exists(data[0] + '/' + date + '/mask'):
        os.makedirs(data[0] + '/' + date + '/mask')
    if not os.path.exists(data[0] + '/' + date + '/output'):
        os.makedirs(data[0] + '/' + date + '/output')

    outpath = data[0] + '/' + date + '/output/' + data[1] + '-out' + str(n) + '.' + outpath_[-1]
    maskpath = data[0] + '/' + date + '/mask/' + data[1] + '-out' + str(n) + '-mask'  + '.' + outpath_[-1]

    cv2.imwrite(outpath, output)
    cv2.imwrite(maskpath, output_mask)

pass
