import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

cv.imshow('Image Display', img)
print(img[0,0,0], img[0,0,1], img[0,0,2])

cv.waitKey()
cv.destroyAllWindows()
