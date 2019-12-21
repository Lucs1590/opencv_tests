import cv2
import utils as ut
import operator
from functools import reduce


class KMeansClass(object):
    def __init__(self):
        ...

    def runKmeans(self, img, rect, k):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        clusters = k

        bar, porcentagens = ut.toKmeans(ut.black_back(img, rect), clusters)
        contaminacao = (porcentagens[-1]*100) / \
            reduce(operator.add, porcentagens)
        print(porcentagens)
        print("Folha + Doença: {}".format(round(reduce(operator.add, porcentagens), 2)))
        print("Contaminação de {}%".format(round(contaminacao, 2)))
        ut.img_plot(bar)
        ut.img_show()
