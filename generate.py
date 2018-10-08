import cv2
import numpy as np
import random

# Path to the background image
background_path = 'data/refugee-camp-before-data.jpg'
# Path to foreground pattern
house_path = 'data/casa5.jpg'
# Mininum and maximum number of rows and columns
# battery_boundaries = [[3, 7], [2, 5]]

# nrows = random.choice(range(battery_boundaries[0][0], battery_boundaries[0][1]+1))
# ncols = random.choice(range(battery_boundaries[1][0], battery_boundaries[1][1]+1))

battery_shape = [3, 7] # shape of the houses disposition
n_houses = 15 # number of houses per battery
n_outputs = 15 # number of house dispositions

background = cv2.imread(background_path)
house = cv2.imread(house_path)


for i in range(n_outputs):

    output = background.copy()
    mask = np.zeros((background.shape[0], background.shape[1], 3), np.uint8)

    possible_positions = []
    for x in range(battery_shape[0]):
        for y in range(battery_shape[1]):
            possible_positions.append([x, y])
    positions = []

    for x in range(n_houses):
        p = random.choice(possible_positions)
        positions.append(p)
        possible_positions.remove(p)

    houses = np.zeros((battery_shape[0]*house.shape[0], battery_shape[1]*house.shape[1], 3), np.uint8)
    _mask = houses.copy()

    for p in positions:
        x = house.shape[0]*p[0]
        y = house.shape[1]*p[1]

        houses[x: x+house.shape[0], y: y+house.shape[1], :] = house
        _mask[x: x+house.shape[0], y: y+house.shape[1], :] = 255

    rotation = random.choice(range(-90, 90))

    # Definir Rotació
    M = cv2.getRotationMatrix2D((0,0), rotation, 1)

    # Definir translació
    M[:,2] = np.array([houses.shape[0], houses.shape[1]])

    # Affine transformation
    rotated = cv2.warpAffine(houses, M, (background.shape[0], background.shape[1]))
    _mask_rotated = cv2.warpAffine(_mask, M, (background.shape[0], background.shape[1]))

    cv2.imshow('output', _mask_rotated)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Getting the part o the image with houses
    x0 = 0
    x1 = _mask_rotated.shape[0]-1

    y0 = 0
    y1 = _mask_rotated.shape[1] - 1
    
    for x in range(x1+1):
        if x0 == 0 and np.sum(_mask_rotated[x]) != 0:
            x0 = x
            
        if x1 == _mask_rotated.shape[0]-1 and np.sum(_mask_rotated[x1-x]) != 0:
            x1 = x1-x
            
        if (x0 != 0 and x1 != _mask_rotated.shape[0]-1):
            break

    for y in range(y1 + 1):
        if np.sum(_mask_rotated[:, y]) != 0 and y0 == 0:
            y0 = y

        if  y1 == _mask_rotated.shape[1] - 1 and np.sum(_mask_rotated[:, y1 - y]) != 0:
            y1 = y1 - y

        if y0 != 0 and y1 != _mask_rotated.shape[1]-1 and x0 != 0 and x1 != _mask_rotated.shape[0]-1:
            break


    cut_rotated = rotated[x0:x1, y0:y1]
    cut__mask = _mask_rotated[x0:x1, y0:y1]

    cv2.imshow('output', cut__mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # start of battery (in pixels)
    start_x = random.choice(range(0, background.shape[0] - cut_rotated.shape[0]))
    start_y = random.choice(range(0, background.shape[1] - cut_rotated.shape[1]))

    mask[start_x:start_x+cut__mask.shape[0], start_y:start_y+cut__mask.shape[1]] = cut__mask
    output[start_x:start_x+cut_rotated.shape[0], start_y:start_y+cut_rotated.shape[1]] = cut_rotated

    ones = np.ones((mask.shape[0], mask.shape[1], 3), np.uint8)*255
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    output[:,:,0] = mask[:,:,0]*output[:,:,0] + (ones[:,:,0] - mask[:,:,0])*background[:,:,0]
    output[:,:,1] = mask[:,:,1]*output[:,:,1] + (ones[:,:,1] - mask[:,:,1])*background[:,:,1]
    output[:,:,2] = mask[:,:,2]*output[:,:,2] + (ones[:,:,2] - mask[:,:,2])*background[:,:,2]

    cv2.imshow('output', output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # output = output * (1-mask/255)
    #
    # cv2.imshow('output', output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    outpath_ = background_path.split('.')
    folder = outpath_[0].split('/')
    print(folder)
    folder[0] + ''
    outpath = ''.join(outpath_[:-1]) + '-out'+str(i) + '.' + outpath_[-1]
    maskpath = ''.join(outpath_[:-1]) + '-out'+ str(i) + '-mask' + '.' + outpath_[-1]
    #cv2.imwrite(outpath, output)
    #cv2.imwrite(maskpath, mask)

pass
