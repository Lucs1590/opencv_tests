import cv2
import numpy as np
import utils as ut
from matplotlib import pyplot as plt


def black_back(img, coord):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, coord, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]
    return img


def color_filter(img, color, diff):
    boundaries_one = [([color[2]-diff, color[1]-diff, color[0]-diff],
                       [color[2]+diff, color[1]+diff, color[0]+diff])]
    for (lower, upper) in boundaries_one:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        ratio_green = cv2.countNonZero(mask)/(img.size/3)
        background = (mask.size - cv2.countNonZero(mask)) / (img.size/3)
        # print('pixels pretos: {}'.format(np.round(background*100, 2)))
        # print('pixels coloridos: {}'.format(np.round(ratio_green*100, 2)))
        # return ut.resize(output)
        return cv2.countNonZero(mask)


img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
rect = (430, 196, 800, 310)

green = [57, 79, 42]
dark_green = [21, 34, 20]
rust = [99, 90, 20]
diff = 20

leaf_area = (color_filter(black_back(img, rect), rust, diff) + color_filter(black_back(
    img, rect), green, diff) + color_filter(black_back(img, rect), dark_green, diff)) / (img.size/3)
percent_leaf = np.round(leaf_area*100, 2)
print(percent_leaf)

# cv2.imshow("Img", )
# cv2.waitKey(0)
