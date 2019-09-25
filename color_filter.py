import numpy as np
import cv2
import utils as ut

img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
green = [57, 79, 42]  # RGB
dark_green = [21, 34, 20]
rust = [99, 90, 20]
diff = 20

# cv2.imshow("images", ut.resize(mask))
cv2.imshow("images", ut.color_filter(img, dark_green, diff))
cv2.waitKey(0)
