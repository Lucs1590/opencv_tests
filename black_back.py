import cv2
import numpy as np
import utils as ut
from matplotlib import pyplot as plt

img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
ut.img_plot(img)
ut.img_show()

rect = (430, 196, 800, 310)
img = ut.black_back(img, rect)


""" cv2.imshow("GrabCut",img)
cv2.waitKey(0) """
plt.imshow(img)
plt.colorbar()
plt.show()
""" cv2.imshow('img',ut.resize(img))
cv2.imwrite('img.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows() """
