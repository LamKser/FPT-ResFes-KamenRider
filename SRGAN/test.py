import cv2
img = cv2.imread("test_img/10.JPG")
img = cv2.resize(img, (img.shape[1]//7, img.shape[0]//7))
cv2.imwrite("resize/abc.jpg", img)