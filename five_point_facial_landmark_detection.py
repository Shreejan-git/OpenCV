import imutils
from imutils import face_utils
import dlib
import cv2
import os
import time

'''
Using HOG- Linear Support Vector Machine for face detection, it took 0.1222 seconds to detect the single face
'''

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

detector = dlib.get_frontal_face_detector() #HOG + LINEAR SVM

predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

img = cv2.imread('images/tilted_head_third.jpg')
#CHECK WHETHER WE NEED A PARTICULAR IMAGE SIZE!
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale frame

start = time.time()
rects = detector(gray, 0) #need to research on 2nd parameter.
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))

'''
boxes = [convert_and_trim_bb(img, r)for r in rects]

for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Output from HOG LSVM", img)
cv2.waitKey(0)
'''

if len(rects) > 0:  # checks whether at least one face is detected
    text = "{} face(s) found".format(len(rects))
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# loop over the face detections
for rect in rects:
    # compute the bounding box of the face and draw it on the frame
    (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
    cv2.rectangle(img, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for (i, (x, y)) in enumerate(shape):
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        
cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


