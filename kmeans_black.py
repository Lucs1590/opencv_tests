import cv2
import numpy as np
import utils as ut
import operator
from matplotlib import pyplot as plt
from functools import reduce
from sklearn.cluster import KMeans


def toKmeans(img, clusters):
    ut.img_plot(img)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=clusters)
    clt.fit(img)
    hist = ut.centroid_histogram(clt)
    bar, porcentagens = ut.plot_colors(hist, clt.cluster_centers_)
    return bar, porcentagens


img = '/home/brito/Documentos/Dev/tcc/resultados/f1_v.png'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rect = (430, 196, 800, 310)
clusters = 5

bar, porcentagens = toKmeans(ut.black_back(img, rect), clusters)
contaminacao = (porcentagens[-1]*100)/reduce(operator.add, porcentagens)
print(porcentagens)
print("Folha + Doença: {}".format(round(reduce(operator.add, porcentagens), 2)))
print("Contaminação de {}%".format(round(contaminacao, 2)))
ut.img_plot(bar)
ut.img_show()
