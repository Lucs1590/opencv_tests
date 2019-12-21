import cv2
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
import operator
import collections
from sklearn.cluster import KMeans


def resize(img, scale_percent=50):
    width = int(img.shape[1] * (scale_percent / 100))
    height = int(img.shape[0] * (scale_percent / 100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def toHSV(img):
    hsv = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)
    return V


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    hist, centroids = organizarCorArray(hist, centroids)
    porcentagens = []
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        if color != [0, 0, 0]:
            porcentagens.append(round(percent*100, 2))
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color, -1)
        startX = endX
    return bar, porcentagens


def img_plot(img):
    plt.figure()
    plt.axis("off")
    plt.imshow(img)


def img_show():
    plt.show()


def black_back(img, coord):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, coord, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    output = img*mask2[:, :, np.newaxis]
    """ cv2.imshow('img',ut.resize(img))
        cv2.imwrite('img.png',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """
    return output
    # return np.hstack([resize(img), resize(output)])


def color_filter(img, color, diff):
    boundaries_one = [([color[2]-diff, color[1]-diff, color[0]-diff],
                       [color[2]+diff, color[1]+diff, color[0]+diff])]

    for (lower, upper) in boundaries_one:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        ratio_green = cv2.countNonZero(mask)/(img.size/3)
        background = (mask.size - cv2.countNonZero(mask)) / (img.size/3)
        print('pixels pretos: {}'.format(np.round(background*100, 2)))
        print('pixels verdes: {}'.format(np.round(ratio_green*100, 2)))
        return resize(output)
        # return np.hstack([ut.resize(img), ut.resize(output)])


def to_Transparent(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("test.png", dst)


def organizarCorArray(hist, centroids):
    aux = {}
    final_val = {}
    cores = []
    percents = []
    for (percent, color) in zip(hist, centroids):
        aux[reduce(operator.add, color.astype(
            "uint8").tolist())] = percent
        final_val[reduce(operator.add, color.astype(
            "uint8").tolist())] = color.astype("uint8").tolist()
    aux = collections.OrderedDict(sorted(aux.items()))
    for k, v in aux.items():
        cores.append(final_val[k])
        percents.append(v)
    return percents, cores


def toKmeans(img, clusters):
    img_plot(img)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=clusters)
    clt.fit(img)
    hist = centroid_histogram(clt)
    bar, porcentagens = plot_colors(hist, clt.cluster_centers_)
    return bar, porcentagens
