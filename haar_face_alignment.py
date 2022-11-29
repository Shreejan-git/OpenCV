import os
import cv2
import numpy as np

frontal_face = os.path.join('opencv-4.6.0','data','haarcascades','haarcascade_frontalface_default.xml')
eye = os.path.join('opencv-4.6.0','data','haarcascades','haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(frontal_face)

eye_cascade = cv2.CascadeClassifier(eye)

original_img = cv2.imread('images/tilted_head.jpg')

img = original_img.copy()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image=gray_img, scaleFactor=1.05, minNeighbors=1, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)

for (x,y,w,h) in faces:
    # cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0),3)
    pass

# cv2.imshow('image', img)
# cv2.waitKey(0)

#SEPERATING THE REASON OF INTEREST (ROI)

roi_gray=gray_img[y:(y+h), x:(x+w)]
roi_color=img[y:(y+h), x:(x+w)]

# print(roi_color)

# cv2.imshow('image', roi_gray)
# cv2.waitKey(0)

#DETECTING THE EYES 

eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
index=0
# Creating for loop in order to divide one eye from another
for (ex , ey,  ew,  eh) in eyes: 
    #ex is x of eyes, ey is y of eyes and so on
   
    if index == 0:
        eye_1 = (ex, ey, ew, eh)
    elif index == 1:
        eye_2 = (ex, ey, ew, eh)
    else:
        print('Detected False positive for the eyes')

# Drawing rectangles around the eyes
    # cv2.rectangle(roi_color, (ex,ey) ,(ex+ew, ey+eh), (0,0,255), 3)
    index = index + 1
  
# cv2.imshow('eyes detection', roi_color)
# cv2.waitKey(0)

if eye_1[0] < eye_2[0]:
   print('here')
   left_eye = eye_1
   right_eye = eye_2
else:
   left_eye = eye_2
   right_eye = eye_1
   
left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
left_eye_x = left_eye_center[0] 
left_eye_y = left_eye_center[1]
 
right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
right_eye_x = right_eye_center[0]
right_eye_y = right_eye_center[1]
 
# cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0) , -1)
# cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0) , -1)
# cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200),3)

if left_eye_y > right_eye_y:
   A = (right_eye_x, left_eye_y)
   # Integer -1 indicates that the image will rotate in the clockwise direction
   direction = -1 
else:
   A = (left_eye_x, right_eye_y)
  # Integer 1 indicates that image will rotate in the counter clockwise  
  # direction
   direction = 1 

# cv2.circle(roi_color, A, 5, (255, 0, 0) , -1)
 
# cv2.line(roi_color,right_eye_center, left_eye_center,(0,200,200),3)
# cv2.line(roi_color,left_eye_center, A,(0,200,200),3)
# cv2.line(roi_color,right_eye_center, A,(0,200,200),3)

#CALCULATING THE ANGLE
delta_x = right_eye_x - left_eye_x
delta_y = right_eye_y - left_eye_y
angle=np.arctan(delta_y/delta_x)
angle = (angle * 180) / np.pi #radian to degree

# Width and height of the image 
h, w = roi_color.shape[:2]
# h, w = img.shape[:2]
# Calculating a center point of the image
# Integer division "//"" ensures that we receive whole numbers
center = (w // 2, h // 2)

# Defining a matrix M and calling
# cv2.getRotationMatrix2D method
M = cv2.getRotationMatrix2D(center, (angle), 0.75)
# Applying the rotation to our image using the
# cv2.warpAffine method
rotated = cv2.warpAffine(roi_color, M, (w, h))
# cv2.imshow('rotated',rotated)
# cv2.imshow('original', original_img)
# cv2.waitKey(0)

join = np.hstack((rotated, roi_color))

# cv2.imwrite('images/final_img.jpg', join)

cv2.imshow('final', join)
cv2.waitKey(0)