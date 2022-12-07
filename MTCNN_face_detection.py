from mtcnn import MTCNN
import cv2
import time
import os
import numpy as np


def align_face_mtcnn(img, left_eye, right_eye):

    desiredLeftEye = (0.39, 0.39)
    desiredFaceWidth = 300
    desiredFaceHeight = None

    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    dist = np.sqrt((delta_x ** 2) + (delta_y ** 2))

    desiredDist = (desiredRightEyeX - desiredLeftEye[0])

    desiredDist *= desiredFaceWidth

    scale = desiredDist / dist

    eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # cv2.circle(img, (eyesCenter[0], eyesCenter[1]), 3, (255, 0, 255), -1)

    M = cv2.getRotationMatrix2D(eyesCenter, (angle), scale)

    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (desiredFaceWidth, desiredFaceHeight)

    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def mtcnn_face_detector_image(originalimg, path, confidence=0.8):
    '''
    The detector returns a list of JSON objects. Each JSON object contains three main keys: 'box', 'confidence' and 'keypoints':

    The bounding box is formatted as [x, y, width, height] under the key 'box'.

    The confidence is the probability for a bounding box to be matching a face.

    The keypoints are formatted into a JSON object with the keys 'left_eye,
    'right_eye', 'nose', 'mouth_left', 'mouth_right'. Each keypoint is identified by a pixel position (x, y).

    Another good example of usage can be found in the file “example.py.” 
    located in the root of this repository. Also, you can run the Jupyter Notebook “example.ipynb” for another example of usage
    '''

    detector = MTCNN()

    img = originalimg
    img = cv2.imread(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = detector.detect_faces(rgb)

    for i in result:
        if i['confidence'] > confidence:  # setting the threshold

            # Uncomment below code to see the B.box of detected face and the landmarks
            x, y, w, h = i['box']
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # c = 0
            # for (x, y) in i['keypoints'].values():
            #     cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            #     cv2.putText(img, str(c + 1), (x - 10, y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            #     c += 1

            left_eye = i['keypoints']['left_eye']
            right_eye = i['keypoints']['right_eye']
            # cv2.line(img, right_eye, left_eye, (67, 67, 67), 2)

            rotated = align_face_mtcnn(
                    img, left_eye, right_eye)  # face alignment
            cv2.imshow('rotated', rotated)
            cv2.imshow('MTCNN output', img)
            cv2.waitKey(0)

        # Below code will give the new image with required area of interest
        img_name = originalimg.split('/')[-1].split('.')[0]
        path = os.path.join(path, f'{img_name+str(np.random.randint(0,100))}.jpg')
        # cv2.imwrite(path, rotated)


def main_fun(base_dir):
    sub_dir = os.listdir(base_dir)
    start = time.time()
    for sub in sub_dir:
        sub_path = os.path.join(base_dir, sub)
        images = os.listdir(sub_path)
        output_dir = os.path.join(sub_path, sub + '_mtcnn')

        for image in images:
            image_path = os.path.join(sub_path, image)

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            if os.path.exists(image_path):
                mtcnn_face_detector_image(
                    image_path, output_dir)  # path and img
            else:
                continue

    end = time.time()
    print('The detection time for MTCNN was', (end-start))


if __name__ == '__main__':

    base_dir = 'celebraty'

    # testing images
    # img = 'images/tilted_head_second.jpg'
    # img = 'images/tilted_head_third.jpg'
    # img = 'images/animated.jpg'
    # img = 'images/double_head.jpg'
    # img = 'images/pryinka_karki5.jpg'
    # mtcnn_face_detector_image(img, 7)

    main_fun(base_dir)
