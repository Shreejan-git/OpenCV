from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread('images/tilted_head.jpg')
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



rects = detector(gray,2)

for rect in rects:
    (x,y,w,h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect) # display the output images
    cv2.imshow('Input image', image)
    cv2.imshow("Original", faceOrig)
    cv2.imshow("Aligned", faceAligned)
    cv2.waitKey(0)
    