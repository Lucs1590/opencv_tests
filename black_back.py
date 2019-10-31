import cv2
import numpy as np
import utils as ut
from matplotlib import pyplot as plt

img = '/home/brito/Documentos/Dev/tcc/img/name182.jpg'
img = cv2.imread(img)
img = ut.resize(img, 32)
ut.img_plot(img)
ut.img_show()

rect = (158, 32, 1023, 526)
img = ut.black_back(img, rect)

""" img = ut.toHSV(img) """


plt.imshow(img)
plt.colorbar()
plt.show()
cv2.imwrite('f19_b.png', img)
""" cv2.imshow('img',ut.resize(img))
cv2.imwrite('img.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows() """
