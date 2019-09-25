import cv2
import numpy as np
import utils as ut
from matplotlib import pyplot as plt

img = '/home/brito/Documentos/Dev/tcc/img/bicho.jpg'
img = cv2.imread(img)
ut.img_plot(img)
ut.img_show()
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (82, 51, 1118, 620)  # (x,y,w,h)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()
""" cv2.imshow('img',ut.resize(img))
cv2.imwrite('img.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows() """
