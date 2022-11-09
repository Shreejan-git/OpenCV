import cv2 

img = cv2.imread('images/treeleaf_task.png')
# edge = cv2.Canny(img, 300,300)
# print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)h = cv
blur = cv2.GaussianBlur(gray.copy(),(3,3), 0)

# blur = cv2.bitwise_not(blur)

_, thrash = cv2.threshold(gray, 200,305, cv2.THRESH_BINARY) #240,255
# canny_ = cv2.Canny(blur,100,100) #180 240

contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #TRee
print(len(contours))

for c in contours:
    approx = cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True),True)
    
    cv2.drawContours(img, [approx],0,(0,150,0),2)
    
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 2:
        # pass
        print(approx)
        cv2.drawContours(img, [approx],0,(255,0,0),0)
        cv2.putText(img, "line", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))



cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#canny vanda threshold halda ramro ayekoxa
#gaussianblur last parameter 0 halda ramro vavyekoxa
# cv2.waitKey(0)
