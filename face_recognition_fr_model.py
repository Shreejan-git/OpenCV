import face_recognition
import cv2
import os
from imutils import paths
import pickle
from retinaface_face_detection import retineface_detector

# base_dir = 'face_for_auto_crop'
base_dir = 'aaa'

# for i in os.listdir(base_dir):
#     img_path = os.path.join(base_dir,i)
    
if os.path.exists(base_dir):
    imagepaths = list(paths.list_images(base_dir))
    
    knownEncodings = []
    knownNames = []
    
    for imgpath in imagepaths:
        print(imgpath)
        name = imgpath.split(os.path.sep)[1].split('_')[0]
        print(name)
        faces_detected = retineface_detector(imgpath)
        for face in faces_detected:
            try:
                cv2.imshow('detected face',face)
                cv2.waitKey()
                face_encodings = face_recognition.face_encodings(face)
                # print(face_encodings)
                knownNames.append(name)
                knownEncodings.append(face_encodings)
            except Exception as e:
                print(face)
    print(knownEncodings)
    
    # print(knownNames)
    print('Serializing the encodings')
        
    data = {'encodings': knownEncodings, 'names':knownNames}
    f = open('encodings.pickle','wb')
    f.write(pickle.dumps(data))
    f.close()
        
    