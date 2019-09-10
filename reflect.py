import cv2
import numpy
import utils as ut

img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
frame = cv2.imread(img)


if frame is None:
    print('Error loading image')
    exit()

rows = frame.shape[0]
cols = frame.shape[1]

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

for i in range(0, cols):
    for j in range(0, rows):
        hsv[j, i][1] = 255

# cv2.imshow("Frame 1", ut.resize(frame))
frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("Frame 2", ut.resize(frame))
cv2.waitKey(30000)
