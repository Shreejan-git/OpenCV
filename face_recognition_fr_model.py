import face_recognition
import cv2
import os
import numpy as np
from imutils import paths
import pickle
from retinaface_face_detection import retineface_detector


def faces_embeddings_database(base_dir):

    if os.path.exists(base_dir):
        # calling the main_fun to retrieve all the images

        imagepaths = list(paths.list_images(base_dir))
        print(f'number of images found {len(imagepaths)}')

        knownEncodings = []
        knownNames = []
        counter = 0
        for imgpath in imagepaths:
            # print(imgpath)
            name = imgpath.split(os.path.sep)[1].split('_')[0]
            # print(name)
            faces_detected = retineface_detector(imgpath)
            if type(faces_detected) is not list:
                print(faces_detected)
            else:
                for face in faces_detected: #type(face) is ndarray                    
                    try:
                        cv2.imshow('detected face',face)
                        cv2.waitKey()
                        face_encodings = face_recognition.face_encodings(face) #type(face_encodings) is list
                        l = len(face_encodings)
                        if l == 0:
                            print(f'this image {imgpath} has a problem')
                            
                        else:
                            knownNames.append(name)
                            knownEncodings.append(face_encodings[0])
                            counter += 1 #to know the exact number of encodings
                        #face_encodings[0] needs to be done orelse list of array will get 
                        #appended instead array only
                    except Exception as e:
                        print("exception being hit",e)

        print('Serializing the encodings')
        data = {'encodings': knownEncodings, 'names': knownNames}
        f = open('embeddings.pickle', 'wb')
        f.write(pickle.dumps(data))
        f.close()
        print(f'Total number of encoding done is {counter}')



def recognize_image_face(image):
    '''
    Original implemetation: https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213   
    '''
    
    data = pickle.loads(open('embeddings.pickle','rb').read())
    
    faces_detected = retineface_detector(image) #type(faces_detected) is list
    if type(faces_detected) is not list:
                print(faces_detected)
    else:
        names = []
        for face in faces_detected: #type(face) is ndarray
            try:
                # cv2.imshow('detected face',face)
                # cv2.waitKey()
                face_encodings = face_recognition.face_encodings(face) #type(face_encodings) is list
                matches = face_recognition.compare_faces(data['encodings'], face_encodings[0],tolerance=0.5) #type(matches) is list
                
                name = 'Unknown! we do not have matching face in our database'
                
                if True in matches:
                    matched_Idxs = [i for (i,b) in enumerate(matches) if b]
                    counts = {}
                    
                    for i in matched_Idxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)
                
                names.append(name) 
                print(names)   

            except Exception as e:
                print(e)

if __name__ == '__main__':
    # base_dir = 'aaa'
    # base_dir = 'face_for_auto_crop'
    # faces_embeddings_database(base_dir)
    img2 = 'images/self11.jpeg'
    img1 = 'images/asha_bhonsle2.jpg'
    # img2 = 'images/ajay_devgan168.jpg'
    recognize_image_face(img2)
