from mtcnn import MTCNN
# from imutils.face_utils import FaceAligner
import cv2
import time
import os
import uuid


def mtcnn_face_detector_image(img, path):
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
    img = cv2.imread(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(rgb)

    for i in result:
        x = i['box'][0]
        y = i['box'][1]
        w = i['box'][2]
        h = i['box'][3]

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # c = 0
        # for (x, y) in i['keypoints'].values():
        #     cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        #     cv2.putText(img, str(c + 1), (x - 10, y - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        #     c += 1

        path = os.path.join(path, f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(path, img)
        # cv2.imshow('MTCNN Face Detector', img)
        # cv2.waitKey(0)


if __name__ == '__main__':

    base_dir = 'celebrity'
    sub_dir = os.listdir(base_dir)
    start = time.time()
    for sub in sub_dir:
        sub_path = os.path.join(base_dir, sub)
        images = os.listdir(sub_path)

        for image in images:
            image_path = os.path.join(sub_path, image)

            output_dir = os.path.join(sub_path, 'mtcnn')

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            if os.path.exists(image_path):
                mtcnn_face_detector_image(
                    image_path, output_dir)  # path and img
            else:
                continue

    end = time.time()
    print('The detection time for MTCNN was', (end-start))
