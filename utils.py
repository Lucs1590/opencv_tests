import cv2

def resize(img, scale_percent=50):
    width = int(img.shape[1] * (scale_percent / 100))
    height = int(img.shape[0] * (scale_percent / 100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

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
