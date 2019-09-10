import numpy as np
import cv2
import utils as ut


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
        print('pixels pretos: {}'.format(np.round(background*100, 2)))
        print('pixels verdes: {}'.format(np.round(ratio_green*100, 2)))
        return ut.resize(output)
        # return np.hstack([ut.resize(img), ut.resize(output)])


img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)

green = [57, 79, 42]  # RGB
dark_green = [21, 34, 20]
rust = [99, 90, 20]
diff = 20

# cv2.imshow("images", ut.resize(mask))
cv2.imshow("images", color_filter(img, dark_green, diff))
cv2.waitKey(0)
