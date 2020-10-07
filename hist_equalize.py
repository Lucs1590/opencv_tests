import cv2

img = cv2.imread("resultados/imagens_oficiais/f1_b.png")

img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_y_cr_cb)

# Applying equalize Hist operation on Y channel.
y_eq = cv2.equalizeHist(y)

img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

cv2.imshow("Original", img)
cv2.imshow("Equalized", img_rgb_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
