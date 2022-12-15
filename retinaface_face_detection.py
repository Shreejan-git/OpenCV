from retinaface import RetinaFace
import cv2
import time
import os
import numpy as np


def retineface_detector(image_path, output_path=None):
    '''
    this function returns the image-array 
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
    print(nfd)

    if nfd == 0:
        return f'No face detected on the given image: {image_path}'
    
    else:
        # for face in faces:
            # cv2.imshow('RetinaFace', face) #this face needs to be send to FR model
            # cv2.waitKey()

            # Below code will give the new cropped and aligned image.
            # img_name = image_path.split('/')[-1].split('.')[0]
            # path = os.path.join(output_path, f'{img_name+str(np.random.randint(0,100))}.jpg')
            # path = 'images/shreejan_test6.jpg'
            # cv2.imwrite(path, face)
            # print('1 face detected on the given image')
            # print(face)
            # return face
        print(f'No of faces detected {nfd}')
        return faces
        
    '''
    elif nfd == 1:
        for face in faces:
            # cv2.imshow('RetinaFace', face) #this face needs to be send to FR model
            # cv2.waitKey()

            # Below code will give the new cropped and aligned image.
            # img_name = image_path.split('/')[-1].split('.')[0]
            # path = os.path.join(output_path, f'{img_name+str(np.random.randint(0,100))}.jpg')
            # path = 'images/shreejan_test6.jpg'
            # cv2.imwrite(path, face)
            print('1 face detected on the given image')
            print(face)
            # return face
    else:
        print(face)
        # if we detect multiple of faces in a single image, each face needs to be handled and sent to FR model for recognition
        return f'Multiple face detected on the given image: {image_path}'
    '''

def main_fun(base_dir):
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

    # image = 'images/netra_test.jpg'
    # image = 'images/tilted_head.jpg'
    image = 'images/self11.jpeg'
    # image = 'images/twogirls.png'
    # image = 'aaa/pravesh_77.jpg'
    # image = 'images/team.jpg'
    for face in retineface_detector(image):
        cv2.imshow('face', face)
        cv2.waitKey()
    # main_fun(base_dir)
