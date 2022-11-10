import cv2
import imutils

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

# cv2.drawContours(img,con,contourIdx=-1, color=(255,0, 0), thickness=0 )
# cv2.imshow('image', img)
# cv2.waitKey(0)

# for i, conn in enumerate(con):

#     if hierarchy[0][i][3] == -1:
#         cv2.drawContours(img, contours=[conn], contourIdx=-1, color=(255,0, 0), thickness=0)
#         cv2.imshow('threshold', img)

#         cv2.waitKey(0)

#     else:
#         print('afdaf')


# for c in con:
#     # print(c)
#     approx = cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True), True)

#     cv2.drawContours(img, [approx], 0, (0, 150, 0), 2)

#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     if len(approx) == 2:

#         # print('h',hierarchy)

#         cv2.drawContours(img, contours=[approx], contourIdx=-1, color=(
#             255, 0, 0), thickness=0)
#         cv2.putText(img, "line", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

#         # print(len(approx))
#         # for a in approx:
#         #     print('a', a)
#         #     print('xy', x, y)
#         #     print()


# cv2.imshow("shapes", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

