import cv2
import numpy as np

img1 = cv2.imread("Happy.png")
img2 = cv2.imread("sunglass_second.jpg")
img2 = cv2.resize(img2, dsize=None, fx=1, fy=0.8)

rows, cols, channels = img2.shape
roi = img1[:rows, :cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

dst = cv2.add(img1_bg, img2_fg)
img1[:rows, :cols] = dst

cv2.imwrite('result.png', img1)