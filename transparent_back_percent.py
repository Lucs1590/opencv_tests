import cv2
import numpy as np
import utils as ut
from matplotlib import pyplot as plt

# Black background
img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (430,196,800,310)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv2.imshow("BlackBack", ut.resize(img))
cv2.waitKey(0)

# Percent of green
green = [57, 79, 42]  # RGB
diff = 25
boundaries = [([green[2]-diff, green[1]-diff, green[0]-diff],
               [green[2]+diff, green[1]+diff, green[0]+diff])]


for (lower, upper) in boundaries:
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    ratio_green = cv2.countNonZero(mask)/(img.size/3)
    print('green pixel percentage:', np.round(ratio_green*100, 2))

    cv2.imshow("Percent of Green", np.hstack([ut.resize(img), ut.resize(output)]))
    cv2.waitKey(0)