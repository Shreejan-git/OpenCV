import cv2
'''
Task 1: Assign the number (1 to 4) below the image of the rectangle with respect to its
length inside the rectangle. The shorter the line lower the number (No need to reorder the
image of the rectangle, only give numbering)

Things I tried to do: 
Since I had to compare the inner line of each rectangle to give them a 
number, I have used ContoursDetection. The line inside also has the sharpe edges and will be 
detected as the seperate contour thats why I was sure that I could detect them seperately on the whole image.
After detecting all the contours of the image including lines, I calculated the area of each contours. Since,
all the lines are smaller than the rectangles and each line has their own area, I sorted them in ascending 
order to get the first 4 smallest contours. And based on that, I gave their corresponding rectangle a number.
'''
img = cv2.imread('images/treeleaf_task.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray.copy(), (3, 3), 0)

# canny_e = cv2.Canny(blur,100,255) #not using canny because it was detecting extra edges inside the rectangles.
# cv2.imshow('canny edge detection', canny_e)

ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
# cv2.imshow('threshold', thresh)

con, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_NONE)

sor = sorted(con, key=cv2.contourArea, reverse=False)

for i, cont in enumerate(sor[:4], 1):
    # cv2.drawContours(img, cont,-1,(0,255,0),1) #drawing on the lines inside the rectangles.
    cv2.putText(img, str(i), (cont[0, 0, 0], cont[0, 0, 1]+61),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # giving the numbers

cv2.imshow('image', img)
cv2.waitKey(0)

