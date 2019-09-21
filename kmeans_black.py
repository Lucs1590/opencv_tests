import cv2
import numpy as np
import utils as ut
import operator
from matplotlib import pyplot as plt
from functools import reduce
from sklearn.cluster import KMeans


def black_back(img, coord):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, coord, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]
    return img


def toKmeans(img, clusters):
    ut.img_plot(img)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(clusters)
    clt.fit(img)
    hist = ut.centroid_histogram(clt)
    bar, porcentagens = ut.plot_colors(hist, clt.cluster_centers_)
    return bar, porcentagens


img = '/home/brito/Documentos/Dev/tcc/img/f1.jpeg'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rect = (430, 196, 800, 310)
clusters = 3

bar, porcentagens = toKmeans(black_back(img, rect), clusters)
print("Folha + Doença: ", reduce(operator.add, porcentagens))
ut.img_plot(bar)
ut.img_show()
