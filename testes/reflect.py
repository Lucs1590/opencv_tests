import cv2
import numpy
import utils as ut

img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)


if img is None:
    print('Error loading image')
    exit()

rect = (430, 196, 800, 310)
img = ut.black_back(img, rect)

rows = img.shape[0]
cols = img.shape[1]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for i in range(0, cols):
    for j in range(0, rows):
        hsv[j, i][1] = 255

# cv2.imshow("img 1", ut.resize(img))
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("img 2", ut.resize(img))
cv2.waitKey(0)
