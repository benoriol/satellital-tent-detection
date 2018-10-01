import cv2
import numpy as np
import random
# KK prova
# Path to the background image
#hola
background_path = 'data/refugee-camp-before-data.jpg'
# Path to foreground pattern
house_path = 'data/casa5.jpg'
# Mininum and maximum number of rows and columns
# battery_boundaries = [[3, 7], [2, 5]]

# nrows = random.choice(range(battery_boundaries[0][0], battery_boundaries[0][1]+1))
# ncols = random.choice(range(battery_boundaries[1][0], battery_boundaries[1][1]+1))


battery_shape = [3, 7]
n_houses = 15
n_outputs = 15

background = cv2.imread(background_path)
house = cv2.imread(house_path)
# soc la mar


for i in range(n_outputs):

    output = background.copy()
    mask = np.zeros((background.shape[0], background.shape[1]), np.uint8)

    start_x = random.choice(range(0, background.shape[0] - battery_shape[0]*house.shape[0]))
    start_y = random.choice(range(0, background.shape[1] - battery_shape[1]*house.shape[1]))

    possible_positions = []
    for x in range(battery_shape[0]):
        for y in range(battery_shape[1]):
            possible_positions.append([x, y])
    positions = []
    for x in range(n_houses):
        p = random.choice(possible_positions)
        positions.append(p)
        possible_positions.remove(p)

    for p in positions:
        x = start_x + house.shape[0]*p[0]
        y = start_y + house.shape[1]*p[1]

        output[x: x+house.shape[0], y: y+house.shape[1]] = house
        mask[x: x + house.shape[0], y: y + house.shape[1]] = 255

        pass
    outpath_ = background_path.split('.')
    outpath = ''.join(outpath_[:-1]) + '-out'+str(i) + '.' + outpath_[-1]
    maskpath = ''.join(outpath_[:-1]) + '-out'+ str(i) + '-mask' + '.' + outpath_[-1]
    cv2.imwrite(outpath, output)
    cv2.imwrite(maskpath, mask)

pass
