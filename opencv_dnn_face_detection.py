import numpy as np
import dlib
import cv2
from imutils import face_utils
import os
import time


def align_face(img, left_eye, right_eye):

    desiredLeftEye = (0.39, 0.39)
    desiredFaceWidth = 300
    desiredFaceHeight = None

    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    # angle = np.degrees(np.arctan2(delta_y, delta_x)) - 180
    # 4-quadrant (range -180 to 180)
    # 2-quadrant inverse function (range -90 to 90)
    angle = np.arctan(delta_y/delta_x)
    # converting the radian into degree
    angle = (angle * 180) / np.pi

    # computing the desired right eye coordinate
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # calculating the distance of the original image
    dist = np.sqrt((delta_x ** 2) + (delta_y ** 2))

    # calculating the distance of the scaled image
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])

    # scales our eye distance based on the desired width.
    desiredDist *= desiredFaceWidth

    scale = desiredDist / dist

    eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (desiredFaceWidth, desiredFaceHeight)

    rotated = cv2.warpAffine(src=img, M=M, dsize=(w, h))

    return rotated


def opencv_dnn_image(configfile, modelfile, original_image, path=5, threshold=0.85):
    modelFile = modelfile
    configFile = configfile

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    image = original_image
    image = cv2.imread(image)
    image = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape[:2]

    # we will have an image of size(h,w,3) but Opencv DNN excepts it to be (1,3,300,300)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(
        300, 300), mean=(104.0, 177.0, 123.0))

    # We can feed the processed image to the caffe model now. This is a basic feed forward step in neural networks.
    net.setInput(blob)
    detections = net.forward()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    for i in detections[0, 0]:
        confidence = i[2]
        if confidence >= threshold:  # making the coordinates DLIB compatible.
            left = (i[3] * w).astype(int)
            top = (i[4] * h).astype(int)
            right = (i[5] * w).astype(int)
            bottom = (i[6] * h).astype(int)

            # cv2.rectangle(image, (left, top),
            #               (right, bottom), (0, 0, 255), 2)

            # text = "{:.2f}%".format(confidence * 100)
            # cv2.putText(image, text, (left, top),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            rect = dlib.rectangle(left, top, right, bottom)
            shape = predictor(gray, rect)
            # print('shape before utils', shape)

            # convert the landmark predictor into (x,y) coordinated in Numpy format.
            shape = face_utils.shape_to_np(shape)
            # print('shape after utils \n', shape)
            # print(shape[0])
            # print(shape[1])

            # for (i, (x, y)) in enumerate(shape):
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            #     cv2.putText(image, str(i + 1), (x - 10, y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["right_eye"]

            # print(lStart, lEnd)
            # print(rStart, rEnd)

            # this will fix which is right and left eye of the input image
            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]
            # print('left eye pts:', leftEyePts)
            # cv2.line(image, rightEyePts[0], leftEyePts[0], (67, 67, 67), 2)

            # print(rightEyePts)

            # I changed here(removed the astype('int'))
            leftEyeCenter = leftEyePts.mean(axis=0)
            # print(leftEyeCenter)
            # cv2.circle(image, (int(leftEyeCenter[0]), int(
            #     leftEyeCenter[1])), 3, (255, 255, 255), -1)
            rightEyeCenter = rightEyePts.mean(
                axis=0)  # and the error was solved
            # cv2.circle(image, (int(rightEyeCenter[0]), int(
            #     rightEyeCenter[1])), 3, (255, 255, 255), -1)

            # aligning the face from a given input image.
            rotated = align_face(image, leftEyeCenter, rightEyeCenter)
            # cv2.imshow("Output OpenCV DNN", image)
            # cv2.imshow('rotated Opencv DNN', rotated)
            # cv2.waitKey(0)

            img_name = original_image.split('/')[-1].split('.')[0]
            path = os.path.join(
                path, f'{img_name+str(np.random.randint(0,100))}.jpg')
            cv2.imwrite(path, rotated)



def main_fun(configFile, modelFile, base_dir):
    sub_dir = os.listdir(base_dir)
    start = time.time()
    for sub in sub_dir:
        sub_path = os.path.join(base_dir, sub)
        images = os.listdir(sub_path)
        output_dir = os.path.join(sub_path, sub + '_DNN')
        
        for image in images:
            image_path = os.path.join(sub_path, image)
            if os.path.isfile(image_path):

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                if os.path.exists(image_path):
                    opencv_dnn_image(configFile, modelFile, image_path, output_dir)
                    
                else:
                    continue

    end = time.time()
    print('The detection time for OpenCV DNN was', (end-start))


if __name__ == '__main__':

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    base = 'celebraties'

    # image = 'images/double_head.jpg'
    # image = 'images/tilted_head.jpg'
    # image = 'images/tilted_head_third.jpg'
    # image = 'images/face.jpg'
    # image = 'images/animated.jpg'
    # image = 'images/a.png'
    image = 'images/pryinka_karki5.jpg'
    image = 'images/abhishek3.jpg'

    # opencv_dnn_image(configFile, modelFile, image)

    main_fun(configFile, modelFile, base)
