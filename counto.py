import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils as ut

# Import the image
img = '/home/brito/Documentos/Dev/tcc/img.jpg'
img = cv2.imread(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 20))
for i in range(0, 3):
    ax = axs[i]
    ax.imshow(img_rgb[:, :, i], cmap = 'gray')
plt.show()

cv2.imshow('img',ut.resize(img_rgb))
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img_rgb)
# plt.colorbar()
# plt.show()