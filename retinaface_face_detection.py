from retinaface import RetinaFace
import cv2
import time
import os
import numpy as np


def retineface_detector(image_path, output_path=None):
    '''
    Original: https://github.com/deepinsight/insightface/tree/master/detection/retinaface
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    # resp = RetinaFace.detect_faces(image)

    # for key in resp.keys():
    #     x,y,w,h = resp[key]['facial_area']
    #     cv2.rectangle(image, (x,y),(w,h), (0,255,0),1)
    # right_eye_x = (resp[key]['landmarks']['right_eye'][0]).astype(int)
    # right_eye_y = (resp[key]['landmarks']['right_eye'][1]).astype(int)

    # left_eye_x = (resp[key]['landmarks']['left_eye'][0]).astype(int)
    # left_eye_y = (resp[key]['landmarks']['left_eye'][1]).astype(int)

    # nose_x = (resp[key]['landmarks']['nose'][0]).astype(int)
    # nose_y = (resp[key]['landmarks']['nose'][1]).astype(int)

    # cv2.circle(image, (right_eye_x, right_eye_y), 2, (0,0,255),-1 )
    # cv2.circle(image, (left_eye_x, left_eye_y), 2, (0,0,255),-1 )
    # cv2.circle(image, (nose_x, nose_y), 2, (0,0,255),-1 )

    faces = RetinaFace.extract_faces(img_path=image, align=True)
    nfd = len(faces)  # number of face detected

    if nfd == 0: #if detector could not detect the image
        return f'No human face detected on the given image: {image_path}'

    else:
        #if output_path is None then its for real-time face verification.
        #else its for building a test-cases.
        if output_path is None:
            print(f'No of faces detected: {nfd} in {image_path} image')
            return faces
        else:
            # Below code will return and save the new cropped and aligned image to the output_path.
            #folder and file, and file's name hirarchy is not dynamic.
            img_name = image_path.split('/')[-1].split('.')[0] 
            # path = 'images/shreejan_test6.jpg'
            for face in faces:
                path = os.path.join(output_path, f'{img_name+str(np.random.randint(0,100))}.jpg')
                cv2.imwrite(path, face)


def main_fun(base_dir):
    '''
    This function is just for building the test cases for multiple images. 
    It provides series of images, residing in the disk, to the retineface_detector 
    function to test the time taken to detect the faces and align them.
    '''
    sub_dir = os.listdir(base_dir)
    start = time.time()
    for sub in sub_dir:
        sub_path = os.path.join(base_dir, sub)
        images = os.listdir(sub_path)
        output_dir = os.path.join(sub_path, sub + '_retinaface')

        for image in images:
            image_path = os.path.join(sub_path, image)

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            if os.path.exists(image_path):
                retineface_detector(
                    image_path, output_dir)  # path and img
            else:
                continue

    end = time.time()
    print('The detection time for RetinaFace was', (end-start))


if __name__ == '__main__':
    base_dir = 'retina'

    image = 'images/asha_bhosle2.jpg'
    for face in retineface_detector(image):
        cv2.imshow('face', face)
        cv2.waitKey()
        
    # main_fun(base_dir)
