import cv2

image = cv2.imread("sample.jpg")

cv2.imshow("Prediction",image)

cv2.waitKey(0)
