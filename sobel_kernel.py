import cv2

img_path = '/home/dell/Documents/handwritten_images/testingimages/deepaksir2.jpg'
gray = cv2.imread(img_path, 0)

choose_kernel = int(input('0: Sobel, 1: Scharr'))
ksize = 3 if choose_kernel == 0 else -1

gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

cv2.namedWindow("Sobel/Scharr X", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sobel/Scharr Y", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sobel/Scharr Combined", cv2.WINDOW_NORMAL)
cv2.namedWindow('gray', cv2.WINDOW_NORMAL)

cv2.imshow('gray', gray)
cv2.imshow("Sobel/Scharr X", gX)
cv2.imshow("Sobel/Scharr Y", gY)
cv2.imshow("Sobel/Scharr Combined", combined)
cv2.waitKey(0)
