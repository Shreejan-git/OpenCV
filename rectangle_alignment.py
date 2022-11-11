import cv2
import numpy as np
'''
Task 2: Align(make the rectangle image straight) all the given images of the rectangle.

Things I tried to do: To make each rectangle straight, first, I tried to detect the four
rectangles on the image by contours detection. My plan was to get the positions/coordinates,
and the angle of each rectangle so that using them I could rotate them first, and then align.
I tried to rotate the specific rectangle inside the image using cv2.getRotationMatrix2D and
cv2.drawContours.

'''

img = cv2.imread('images/treeleaf_task.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray, (5, 5), 0)

ret, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
    thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_d = img.copy()

output = np.zeros(img.shape[:2])

for i, con in enumerate(contours):

    if hierarchy[0][i][3] == 0: #only getting the contours of the rectangles.
        print(i)

        rect = cv2.minAreaRect(con)
        # print(rect)
        # trying to get the (x,y),(height, width) and angle of each rectangle.

        (x, y), (height, width), angle = rect
        # x and y give us the center point of every rectangle

        box = cv2.boxPoints(rect).astype(int)
        print(box)
        # this will return the coordinates of the rectangles

        height, width = height//2, width//2

        R = cv2.getRotationMatrix2D((x, y), 90, 1) 

        con = cv2.warpAffine(con, R, dsize=(int(height),int(width)))
        
        # cv2.rectangle(output,(255,110),(750,210),(1,1,1),-1 )
        
        #I tried to apply the rotation but could not do it.
        

        cv2.imshow('original img', img_d[291:370])
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
