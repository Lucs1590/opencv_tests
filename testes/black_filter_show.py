import cv2
import numpy as np
import utils as ut
from matplotlib import pyplot as plt

img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
rect = (430, 196, 800, 310)

green = [57, 79, 42]  # RGB
dark_green = [21, 34, 20]
rust = [99, 90, 20]
diff = 20

cv2.imshow("Img", ut.color_filter(ut.black_back(img, rect), rust, diff))
cv2.waitKey(0)
