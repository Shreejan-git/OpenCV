import numpy as np
import dlib
import cv2
import imutils
from imutils import face_utils

def opencv_dnn_image(configfile, modelfile, image, threshold=0.9):
    modelFile = modelfile
    configFile = configfile

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    image = cv2.imread(image)
    image = cv2.resize(image,(300,300))
    # print(image.shape)
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # we will have an image of size(h,w,3) but Opencv DNN excepts it to be (1,3,300,300)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    # print(blob.shape)

    # We can feed the processed image to the caffe model now. This is a basic feed forward step in neural networks.
    net.setInput(blob)
    detections = net.forward()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')


    for i in detections[0, 0]:
        confidence = i[2]
        if confidence > threshold:
            left = (i[3] * 300).astype(int)
            top = (i[4] * 300).astype(int)
            right = (i[5] * 300).astype(int)
            bottom = (i[6] * 300).astype(int)
            cv2.rectangle(image, (left, top+10),
                          (right, bottom), (0, 0, 255), 2)
            text = "{:.2f}%".format(confidence * 100)
            cv2.putText(image, text, (left, top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            rect = dlib.rectangle(left, top, right, bottom)
            # print(rect)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (i, (x, y)) in enumerate(shape):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(image, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            


    # show the output image
    cv2.imshow("Output OpenCV DNN", image)
    cv2.waitKey(0)
    





if __name__ == '__main__':

    image = 'images/double_head.jpg'
    # image = 'images/tilted_head.jpg'

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"

    opencv_dnn_image(configFile, modelFile, image)
