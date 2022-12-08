import mediapipe as mp
import cv2
import numpy as np
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


def mediapipe_face_detection(originalimg, path, confidence=0.8):
    
    img = originalimg
    img = cv2.imread(img)

    mp_face_detection = mp.solutions.face_detection
    
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence = confidence)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img)
    print(results.detections)

    if results.detections:
        for face in results.detections:
            confidence = face.score
            bounding_box = face.location_data.relative_bounding_box
            
            x = int(bounding_box.xmin * img.shape[1])
            w = int(bounding_box.width * img.shape[1])
            y = int(bounding_box.ymin * img.shape[0])
            h = int(bounding_box.height * img.shape[0])
            
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness = 2)
            
            landmarks = face.location_data.relative_keypoints
    
            right_eye = (int(landmarks[0].x * img.shape[1]), int(landmarks[0].y * img.shape[0]))
            left_eye = (int(landmarks[1].x * img.shape[1]), int(landmarks[1].y * img.shape[0]))
            
            cv2.circle(img, right_eye, 2, (0, 0, 255), -1)
            cv2.circle(img, left_eye, 2, (0, 0, 255), -1)
            
            rotated = align_face(img, left_eye, right_eye)

            # img_name = originalimg.split('/')[-1].split('.')[0]
            # path = os.path.join(path, f'{img_name+str(np.random.randint(0,100))}.jpg')
            # cv2.imwrite(path, rotated)
            cv2.imshow('original image', img)
            cv2.imshow('Media Pipe Image', rotated)
            cv2.waitKey()
            
            
def main_fun(base_dir):
    sub_dir = os.listdir(base_dir)
    start = time.time()
    for sub in sub_dir:
        sub_path = os.path.join(base_dir, sub)
        images = os.listdir(sub_path)
        output_dir = os.path.join(sub_path, sub + '_mediapipe')
        
        for image in images:
            image_path = os.path.join(sub_path, image)
            if os.path.isfile(image_path):

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                if os.path.exists(image_path):
                    mediapipe_face_detection(image_path, output_dir)
                    
                else:
                    continue

    end = time.time()
    print('The detection time for MediaPipe was', (end-start))        
            
            
if __name__ == '__main__':

    base_dir = 'celebraty'

    # testing images
    # img = 'images/tilted_head_second.jpg'
    # img = 'images/tilted_head_third.jpg'
    # img = 'images/animated.jpg'
    # img = 'images/double_head.jpg'
    img = 'images/pryinka_karki5.jpg'
    # img = 'images/burqa/jpg'
    img = 'images/burqa.jpg'
    img = 'images/team1.jpg'
    # img = 'images/twogirls.png'
    mediapipe_face_detection(img, 7)

    # main_fun(base_dir)