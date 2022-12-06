import numpy as np
import dlib
import cv2
from imutils import face_utils


def align_face(img, left_eye, right_eye):

    desiredLeftEye = (0.35, 0.35)
    desiredFaceWidth = 300
    desiredFaceHeight = None

    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    # angle = np.degrees(np.arctan2(delta_x, delta_y)) - 180
    angle = np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    dist = np.sqrt((delta_x ** 2) + (delta_y ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    
    print(eyesCenter)

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (desiredFaceWidth, desiredFaceHeight)

    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def opencv_dnn_image(configfile, modelfile, image, threshold=0.9):
    modelFile = modelfile
    configFile = configfile

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    image = cv2.imread(image)
    # image = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (h, w) = image.shape[:2]

    # we will have an image of size(h,w,3) but Opencv DNN excepts it to be (1,3,300,300)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    # We can feed the processed image to the caffe model now. This is a basic feed forward step in neural networks.
    net.setInput(blob)
    detections = net.forward()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    for i in detections[0, 0]:
        confidence = i[2]
        if confidence > threshold:
            left = (i[3] * w).astype(int)
            top = (i[4] * h).astype(int)
            right = (i[5] * w).astype(int)
            bottom = (i[6] * h).astype(int)
        
            cv2.rectangle(image, (left, top+10),
                          (right, bottom), (0, 0, 255), 2)
            
            # text = "{:.2f}%".format(confidence * 100)
            # cv2.putText(image, text, (left, top),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            rect = dlib.rectangle(left, top, right, bottom)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # for (i, (x, y)) in enumerate(shape):
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            #     cv2.putText(image, str(i + 1), (x - 10, y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["right_eye"]

            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]

            # I changed here(removed the astype('int'))
            leftEyeCenter = leftEyePts.mean(axis=0)
            rightEyeCenter = rightEyePts.mean(
                axis=0)  # and the error was solved

            # aligning the face from a given input image.
            rotated = align_face(image, leftEyeCenter, rightEyeCenter)
            cv2.imshow("Output OpenCV DNN", image)
            cv2.imshow('rotated Opencv DNN', rotated)
            cv2.waitKey(0)


if __name__ == '__main__':

    # image = 'images/double_head.jpg'
    image = 'images/tilted_head.jpg'

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"

    opencv_dnn_image(configFile, modelFile, image)
