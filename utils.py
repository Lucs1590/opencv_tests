import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize(img, scale_percent=50):
    width = int(img.shape[1] * (scale_percent / 100))
    height = int(img.shape[0] * (scale_percent / 100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def toSaturation(img):
    hue, saturation, value = cv2.split(img)
    return saturation


def toThreshold(sat):
    retval, thresholded = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def toMedian(thresh):
    medianFiltered = cv2.medianBlur(thresh, 5)
    return medianFiltered


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    porcentagens = []
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        if color.astype("uint8").tolist() != [0, 0, 0]:
            porcentagens.append(round(percent*100, 2))
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    return bar, porcentagens


def img_plot(img):
    plt.figure()
    plt.axis("off")
    plt.imshow(img)


def img_show():
    plt.show()
