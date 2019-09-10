import numpy as np
import cv2.cv2 as cv2
img = '/home/brito/Documentos/Dev/tcc/img/cercosporiose/IMG_20190409_202040.jpg'


def lerImg(img):
    return cv2.imread(img)


def toHSV(img):
    hsv = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv


def toSaturation(img):
    hue, saturation, value = cv2.split(hsv)
    return saturation


def toThreshold(sat):
    retval, thresholded = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def toMedian(thresh):
    medianFiltered = cv2.medianBlur(thresh, 5)
    return medianFiltered


def createControl():
    cv2.namedWindow('controlador', 0)
    cv2.createTrackbar('h min', 'controlador', 0, 255, update)
    cv2.createTrackbar('h max', 'controlador', 0, 255, update)
    cv2.createTrackbar('s min', 'controlador', 0, 255, update)
    cv2.createTrackbar('s max', 'controlador', 0, 255, update)
    cv2.createTrackbar('v min', 'controlador', 0, 255, update)
    cv2.createTrackbar('v max', 'controlador', 0, 255, update)
    im = cv2.resize(src, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)


def filterOne(img):
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(img, kernel, iterations=3)
    dilate = cv2.dilate(erode, kernel, iterations=3)
    gaussian = cv2.GaussianBlur(dilate, (5, 5), 1)
    final = cv2.erode(gaussian, np.ones((3, 3), np.uint8), iterations=5)
    return final


img = lerImg(img)
hsv = toHSV(img)
saturacao = toSaturation(hsv)
thresh = toThreshold(saturacao)
median = toMedian(thresh)
filter_kernel = filterOne(thresh)
filter_kernel = filterOne(median)

array_contorno, hierarchy = cv2.findContours(
    median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
area_total = 0

for contorno in array_contorno:
    M = cv2.moments(contorno)
    cX = int(M["m10"] / M["m00"]) if int(M["m10"]) else 0
    cY = int(M["m01"] / M["m00"]) if int(M["m01"]) else 0

    first = cv2.drawContours(img, [contorno], -1, (0, 255, 0), 2)
    second = cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)

    area = cv2.contourArea(contorno)
    area_total += area


def resize(img, scale_percent=20):
    width = int(img.shape[1] * (scale_percent / 100))
    height = int(img.shape[0] * (scale_percent / 100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized



cv2.imshow('Image1', resize(median))
cv2.imshow('Image2', resize(thresh))
# cv2.imshow('Image2', resize(first))
print(area_total)
# print(media)
cv2.waitKey(25000)
